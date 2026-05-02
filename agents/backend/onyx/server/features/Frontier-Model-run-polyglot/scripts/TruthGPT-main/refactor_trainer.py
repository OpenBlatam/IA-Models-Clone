import re
import sys

file_path = r"c:\blatam-academy\agents\backend\onyx\server\features\Frontier-Model-run-polyglot\scripts\TruthGPT-main\optimization_core\trainers\trainer.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Remove the old dataclass
pattern_dataclass = re.compile(r"@dataclass\nclass TrainerConfig:.*?resume_checkpoint_dir:\s*Optional\[str\]\s*=\s*None\n", re.DOTALL)
code = pattern_dataclass.sub("from optimization_core.trainers.config import TrainerConfig\n", code)

# Substitution maps
subs = {
    # Model
    r"\bcfg\.model_name\b": "cfg.model.name_or_path",
    r"\bself\.cfg\.model_name\b": "self.cfg.model.name_or_path",
    r"\bcfg\.gradient_checkpointing\b": "cfg.model.gradient_checkpointing",
    r"\bself\.cfg\.gradient_checkpointing\b": "self.cfg.model.gradient_checkpointing",
    r"\bcfg\.lora_enabled\b": "cfg.model.lora_enabled",
    r"\bcfg\.lora_r\b": "cfg.model.lora_r",
    r"\bcfg\.lora_alpha\b": "cfg.model.lora_alpha",
    r"\bcfg\.lora_dropout\b": "cfg.model.lora_dropout",
    
    # Training
    r"\bcfg\.epochs\b": "cfg.training.epochs",
    r"\bself\.cfg\.epochs\b": "self.cfg.training.epochs",
    r"\bcfg\.train_batch_size\b": "cfg.training.train_batch_size",
    r"\bself\.cfg\.train_batch_size\b": "self.cfg.training.train_batch_size",
    r"\bcfg\.eval_batch_size\b": "cfg.training.eval_batch_size",
    r"\bself\.cfg\.eval_batch_size\b": "self.cfg.training.eval_batch_size",
    r"\bcfg\.grad_accum_steps\b": "cfg.training.grad_accum_steps",
    r"\bself\.cfg\.grad_accum_steps\b": "self.cfg.training.grad_accum_steps",
    r"\bcfg\.max_grad_norm\b": "cfg.training.max_grad_norm",
    r"\bself\.cfg\.max_grad_norm\b": "self.cfg.training.max_grad_norm",
    r"\bcfg\.learning_rate\b": "cfg.training.learning_rate",
    r"\bself\.cfg\.learning_rate\b": "self.cfg.training.learning_rate",
    r"\bcfg\.weight_decay\b": "cfg.training.weight_decay",
    r"\bself\.cfg\.weight_decay\b": "self.cfg.training.weight_decay",
    r"\bcfg\.warmup_ratio\b": "cfg.training.warmup_ratio",
    r"\bcfg\.scheduler\b": "cfg.training.scheduler",
    r"\bcfg\.mixed_precision\b": "cfg.training.mixed_precision",
    r"\bself\.cfg\.mixed_precision\b": "self.cfg.training.mixed_precision",
    r"\bcfg\.early_stopping_patience\b": "cfg.training.early_stopping_patience",
    r"\bself\.cfg\.early_stopping_patience\b": "self.cfg.training.early_stopping_patience",
    r"\bcfg\.log_interval\b": "cfg.training.log_interval",
    r"\bself\.cfg\.log_interval\b": "self.cfg.training.log_interval",
    r"\bcfg\.eval_interval\b": "cfg.training.eval_interval",
    r"\bself\.cfg\.eval_interval\b": "self.cfg.training.eval_interval",
    r"\bcfg\.select_best_by\b": "cfg.training.select_best_by",
    r"\bself\.cfg\.select_best_by\b": "self.cfg.training.select_best_by",
    
    # Hardware
    r"\bcfg\.device\b": "cfg.hardware.device",
    r"\bself\.cfg\.device\b": "self.cfg.hardware.device",
    r"\bcfg\.multi_gpu\b": "cfg.hardware.multi_gpu",
    r"\bcfg\.ddp\b": "cfg.hardware.ddp",
    r"\bcfg\.allow_tf32\b": "cfg.hardware.allow_tf32",
    r"\bcfg\.torch_compile\b": "cfg.hardware.torch_compile",
    r"\bcfg\.compile_mode\b": "cfg.hardware.compile_mode",
    r"\bcfg\.fused_adamw\b": "cfg.hardware.fused_adamw",
    r"\bcfg\.detect_anomaly\b": "cfg.hardware.detect_anomaly",
    r"\bself\.cfg\.detect_anomaly\b": "self.cfg.hardware.detect_anomaly",
    r"\bcfg\.use_profiler\b": "cfg.hardware.use_profiler",
    r"\bself\.cfg\.use_profiler\b": "self.cfg.hardware.use_profiler",
    r"\bcfg\.num_workers\b": "cfg.hardware.num_workers",
    r"\bself\.cfg\.num_workers\b": "self.cfg.hardware.num_workers",
    r"\bcfg\.prefetch_factor\b": "cfg.hardware.prefetch_factor",
    r"\bself\.cfg\.prefetch_factor\b": "self.cfg.hardware.prefetch_factor",
    r"\bcfg\.persistent_workers\b": "cfg.hardware.persistent_workers",
    r"\bself\.cfg\.persistent_workers\b": "self.cfg.hardware.persistent_workers",
    
    # Checkpoint
    r"\bcfg\.ckpt_interval_steps\b": "cfg.checkpoint.interval_steps",
    r"\bself\.cfg\.ckpt_interval_steps\b": "self.cfg.checkpoint.interval_steps",
    r"\bcfg\.ckpt_keep_last\b": "cfg.checkpoint.keep_last",
    r"\bself\.cfg\.ckpt_keep_last\b": "self.cfg.checkpoint.keep_last",
    r"\bcfg\.save_safetensors\b": "cfg.checkpoint.save_safetensors",
    r"\bcfg\.resume_from_checkpoint\b": "cfg.checkpoint.resume_from_checkpoint",
    r"\bself\.cfg\.resume_from_checkpoint\b": "self.cfg.checkpoint.resume_from_checkpoint",
    r"\bcfg\.resume_enabled\b": "cfg.checkpoint.resume_enabled",
    r"\bcfg\.resume_checkpoint_dir\b": "cfg.checkpoint.resume_checkpoint_dir",
    
    # EMA
    r"\bcfg\.ema_enabled\b": "cfg.ema.enabled",
    r"\bself\.cfg\.ema_enabled\b": "self.cfg.ema.enabled",
    r"\bcfg\.ema_decay\b": "cfg.ema.decay",
    r"\bself\.cfg\.ema_decay\b": "self.cfg.ema.decay",
}

for pattern, repl in subs.items():
    code = re.sub(pattern, repl, code)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Trainer refactoring completed.")
