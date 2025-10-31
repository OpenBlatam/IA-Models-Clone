from typing import Any, Dict


class Callback:
    def on_log(self, state: Dict[str, Any]) -> None:
        pass

    def on_eval(self, state: Dict[str, Any]) -> None:
        pass

    def on_save(self, state: Dict[str, Any]) -> None:
        pass


class PrintLogger(Callback):
    def on_log(self, state: Dict[str, Any]) -> None:
        step = state.get("step")
        msg = state.get("message", "")
        print(f"[log] step={step} {msg}")

    def on_eval(self, state: Dict[str, Any]) -> None:
        step = state.get("step")
        val_loss = state.get("val_loss")
        improved = state.get("improved")
        print(f"[eval] step={step} val_loss={val_loss:.4f} improved={improved}")

    def on_save(self, state: Dict[str, Any]) -> None:
        path = state.get("path")
        print(f"[save] checkpoint -> {path}")


class WandbLogger(Callback):
    def __init__(self, project: str, run_name: str) -> None:
        try:
            import wandb
            self._wandb = wandb
            if not wandb.run:
                wandb.init(project=project or "truthgpt", name=run_name or None)
        except Exception:
            self._wandb = None

    def on_log(self, state):
        if self._wandb is None:
            return
        data = {}
        msg = state.get("message", "")
        for part in msg.split():
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    data[k] = float(v)
                except Exception:
                    data[k] = v
        step = state.get("step")
        self._wandb.log(data, step=step)

    def on_eval(self, state):
        if self._wandb is None:
            return
        self._wandb.log({"val_loss": state.get("val_loss")}, step=state.get("step"))

    def on_save(self, state):
        pass


class TensorBoardLogger(Callback):
    def __init__(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir or "runs")
        except Exception:
            self._writer = None

    def on_log(self, state):
        if self._writer is None:
            return
        step = state.get("step")
        msg = state.get("message", "")
        # Parse known keys
        metrics = {}
        for part in msg.split():
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    metrics[k] = float(v)
                except Exception:
                    continue
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, global_step=step)

    def on_eval(self, state):
        if self._writer is None:
            return
        self._writer.add_scalar("val/loss", float(state.get("val_loss", 0.0)), global_step=state.get("step"))

    def on_save(self, state):
        pass


