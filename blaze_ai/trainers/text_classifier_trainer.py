from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from contextlib import nullcontext
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils.early_stopping import EarlyStopping, EarlyStoppingConfig
from ..utils.metrics import classification_metrics
from ..utils.checkpointing import save_checkpoint, load_checkpoint
from ..utils.logging import get_logger
from ..utils.experiment import ExperimentTracker
from ..utils.device import autocast_context, move_batch_to_device

@dataclass
class TrainerConfig:
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    grad_clip_value: float | None = None
    use_fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    reduce_on_plateau: bool = True
    reduce_factor: float = 0.5
    reduce_patience: int = 1
    log_every_steps: int = 50
    debug_detect_anomaly: bool = False
    log_learning_rate: bool = True
    use_data_parallel: bool = False
    use_distributed: bool = False
    ddp_backend: str = "nccl"
    ddp_init_method: str = "env://"
    ddp_find_unused_parameters: bool = False
    gradient_accumulation_steps: int = 1
    output_dir: str = "./outputs/blaze_ai"
    save_best_only: bool = True
    save_every_epochs: int = 0
    use_tensorboard: bool = False
    tensorboard_run: str = "runs/blaze_ai"
    use_wandb: bool = False
    wandb_project: str = "blaze_ai"
    wandb_run_name: str = "text_classifier"
    resume_from: str | None = None
    save_last: bool = True


class TextClassifierTrainer:
    def __init__(self, model: nn.Module, config: Optional[TrainerConfig] = None) -> None:
        self.config = config or TrainerConfig()
        device_str = self.config.device
        self.is_distributed = bool(self.config.use_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1)
        self.is_data_parallel = bool(self.config.use_data_parallel and not self.is_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1)

        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            if not dist.is_available():
                raise RuntimeError("torch.distributed not available")
            if not dist.is_initialized():
                dist.init_process_group(backend=self.config.ddp_backend, init_method=self.config.ddp_init_method)
            device_str = f"cuda:{local_rank}"
            model = model.to(device_str)
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=self.config.ddp_find_unused_parameters)
        else:
            model = model.to(device_str)
            if self.is_data_parallel:
                self.model = nn.DataParallel(model)
            else:
                self.model = model
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_fp16 and self.config.device.startswith("cuda"))
        self.logger = get_logger(__name__)
        # Experiment tracker
        self.tracker = ExperimentTracker(self.config.output_dir)
        # Save config snapshot
        try:
            self.tracker.log_config({"trainer": self.config.__dict__})
        except Exception:
            pass

    def _create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        self.model.train()
        optimizer = self._create_optimizer()
        total_steps = len(train_loader) * self.config.epochs
        if self.config.reduce_on_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.config.reduce_factor, patience=self.config.reduce_patience
            )
        else:
            scheduler = self._create_scheduler(optimizer, total_steps)
        criterion = nn.CrossEntropyLoss()
        stopper = EarlyStopping(
            EarlyStoppingConfig(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode="min",
            )
        ) if self.config.early_stopping else None

        global_step = 0
        grad_accum_steps = max(1, int(self.config.gradient_accumulation_steps))
        micro_step = 0
        # Resume from checkpoint if provided
        if self.config.resume_from:
            try:
                state = load_checkpoint(self.config.resume_from, self.model, optimizer, scheduler=None, scaler=self.scaler)
                global_step = int(state.get("global_step", 0))
                self.logger.info("resumed_from path=%s step=%d", self.config.resume_from, global_step)
            except Exception:
                self.logger.warning("failed_to_resume path=%s", self.config.resume_from)
        # Tracking setup
        tb_writer = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                tb_writer = SummaryWriter(self.config.tensorboard_run)
            except Exception:
                self.logger.warning("TensorBoard not available")
        if self.config.use_wandb:
            try:
                import wandb  # type: ignore
                wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name)
                wandb.config.update({"epochs": self.config.epochs, "lr": self.config.learning_rate})
            except Exception:
                self.logger.warning("W&B not available or failed to init")
        for epoch in range(self.config.epochs):
            self.logger.info("epoch_start epoch=%d/%d steps=%d", epoch + 1, self.config.epochs, len(train_loader))
            for batch in train_loader:
                try:
                    batch = move_batch_to_device(batch, self.config.device)
                    anomaly_ctx = torch.autograd.detect_anomaly() if self.config.debug_detect_anomaly else nullcontext()
                    with anomaly_ctx:
                        with autocast_context(self.config.use_fp16, self.config.device):
                            logits = self.model(batch["input_ids"], batch.get("attention_mask"))
                            loss = criterion(logits, batch["labels"])
                            if grad_accum_steps > 1:
                                loss = loss / grad_accum_steps
                        # Drop non-finite batches to keep training stable
                        if not torch.isfinite(loss):
                            self.logger.warning("non_finite_loss epoch=%d step=%d", epoch + 1, global_step)
                            optimizer.zero_grad(set_to_none=True)
                            continue
                        self.scaler.scale(loss).backward()
                    micro_step += 1
                    do_step = (micro_step % grad_accum_steps == 0)
                    if do_step:
                        # Unscale before clipping
                        self.scaler.unscale_(optimizer)
                        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        if self.config.grad_clip_value is not None:
                            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_clip_value)

                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        if self.config.reduce_on_plateau:
                            # will be stepped after validation
                            pass
                        else:
                            scheduler.step()
                        global_step += 1

                        if self.config.log_every_steps and global_step % self.config.log_every_steps == 0:
                            lr = optimizer.param_groups[0].get("lr", 0.0) if self.config.log_learning_rate else 0.0
                            self.logger.info(
                                "train step=%d epoch=%d loss=%.6f lr=%.6f",
                                global_step,
                                epoch + 1,
                                float(loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1.0)),
                                float(lr),
                            )
                            # Tracker
                            try:
                                self.tracker.log_metrics({"train/loss": float(loss.item()), "train/lr": float(lr)}, step=global_step)
                            except Exception:
                                pass
                            if tb_writer is not None:
                                tb_writer.add_scalar("train/loss", float(loss.item()), global_step)
                                tb_writer.add_scalar("train/lr", float(lr), global_step)
                            try:
                                import wandb  # type: ignore
                                if self.config.use_wandb:
                                    wandb.log({"train/loss": float(loss.item()), "train/lr": float(lr), "step": global_step})
                            except Exception:
                                pass
                except Exception as exc:
                    self.logger.exception("train_step_error epoch=%d step=%d", epoch + 1, global_step)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                if self.config.reduce_on_plateau:
                    scheduler.step(metrics["val_loss"])  # type: ignore[arg-type]
                self.logger.info("val epoch=%d loss=%.6f acc=%.4f", epoch + 1, float(metrics["val_loss"]), float(metrics.get("accuracy", 0.0)))
                try:
                    self.tracker.log_metrics({"val/loss": float(metrics["val_loss"]), "val/accuracy": float(metrics.get("accuracy", 0.0))}, step=global_step)
                except Exception:
                    pass
                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss", float(metrics["val_loss"]), global_step)
                    tb_writer.add_scalar("val/accuracy", float(metrics.get("accuracy", 0.0)), global_step)
                try:
                    import wandb  # type: ignore
                    if self.config.use_wandb:
                        wandb.log({"val/loss": float(metrics["val_loss"]), "val/accuracy": float(metrics.get("accuracy", 0.0)), "step": global_step})
                except Exception:
                    pass
                # Checkpointing
                ckpt_path = None
                if self.config.save_every_epochs and (epoch + 1) % self.config.save_every_epochs == 0:
                    ckpt_path = f"{self.config.output_dir}/checkpoint_epoch{epoch+1}.pt"
                    save_checkpoint(ckpt_path, self.model, optimizer, scheduler, self.scaler, epoch + 1, global_step)
                if self.config.save_best_only:
                    ckpt_path = f"{self.config.output_dir}/best.pt"
                    save_checkpoint(ckpt_path, self.model, optimizer, scheduler, self.scaler, epoch + 1, global_step, extra={"val_loss": float(metrics["val_loss"])})
                if self.config.save_last:
                    save_checkpoint(f"{self.config.output_dir}/last.pt", self.model, optimizer, scheduler, self.scaler, epoch + 1, global_step)
                if stopper is not None and stopper.step(metrics["val_loss"]):
                    self.logger.info("early_stop epoch=%d step=%d val_loss=%.6f", epoch + 1, global_step, float(metrics["val_loss"]))
                    break
            else:
                metrics = {"val_accuracy": float("nan"), "val_loss": float("nan")}

        if tb_writer is not None:
            try:
                tb_writer.flush()
                tb_writer.close()
            except Exception:
                pass
        return {"global_step": global_step, **metrics}

    @torch.inference_mode()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for batch in data_loader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            logits = self.model(batch["input_ids"], batch.get("attention_mask"))
            loss = criterion(logits, batch["labels"])
            total_loss += float(loss.item()) * batch["labels"].size(0)
            all_logits.append(logits)
            all_labels.append(batch["labels"])
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        avg_loss = total_loss / max(1, labels_cat.numel())
        cls_metrics = classification_metrics(logits_cat, labels_cat, average="macro")
        return {"val_loss": avg_loss, **cls_metrics}


