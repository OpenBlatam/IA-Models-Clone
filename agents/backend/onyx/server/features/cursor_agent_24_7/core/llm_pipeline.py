"""
LLM Pipeline - Pipeline avanzado para procesamiento con LLMs
============================================================

Pipeline profesional para procesamiento de texto con modelos de transformers.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuración para modelos LLM"""
    model_name: str = "gpt2"
    device: str = "auto"  # auto, cpu, cuda
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_cache: bool = True
    pad_token_id: Optional[int] = None


class LLMPipeline:
    """Pipeline para procesamiento con LLMs"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialized = False
        
    async def initialize(self):
        """Inicializar modelo y tokenizer"""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Determinar dispositivo
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            logger.info(f"Loading model {self.config.model_name} on {self.device}")
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Configurar pad_token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.config.pad_token_id is None:
                    self.config.pad_token_id = self.tokenizer.eos_token_id
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            
            logger.info(f"✅ Model {self.config.model_name} loaded successfully")
            
        except ImportError:
            logger.warning("transformers not available")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """Generar texto a partir de un prompt"""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Usar valores de configuración o parámetros
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        # Tokenizar
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Generar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=self.config.use_cache,
                **kwargs
            )
        
        # Decodificar
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover el prompt original
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def encode(self, text: str) -> torch.Tensor:
        """Codificar texto a embeddings"""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Usar el último hidden state del último token
            embeddings = outputs.hidden_states[-1][0, -1, :]
        
        return embeddings
    
    def classify(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Clasificar texto usando el modelo"""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        # Crear prompts para cada label
        scores = {}
        for label in labels:
            prompt = f"{text}\nLabel: {label}"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usar logits del último token
                logits = outputs.logits[0, -1, :]
                # Score como probabilidad
                score = torch.softmax(logits, dim=-1).max().item()
                scores[label] = score
        
        # Normalizar scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def complete_code(self, code: str, language: str = "python") -> str:
        """Completar código"""
        prompt = f"# {language}\n{code}\n"
        return self.generate(prompt, max_new_tokens=200, temperature=0.3)
    
    def explain_code(self, code: str) -> str:
        """Explicar código"""
        prompt = f"Explain this code:\n```\n{code}\n```\nExplanation:"
        return self.generate(prompt, max_new_tokens=150, temperature=0.5)
    
    def fix_code(self, code: str, error: Optional[str] = None) -> str:
        """Corregir código"""
        if error:
            prompt = f"Fix this code that has error '{error}':\n```\n{code}\n```\nFixed code:\n```\n"
        else:
            prompt = f"Fix this code:\n```\n{code}\n```\nFixed code:\n```\n"
        
        return self.generate(prompt, max_new_tokens=200, temperature=0.2)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """Resumir texto"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}\n\nSummary:"
        return self.generate(prompt, max_new_tokens=max_length, temperature=0.5)
    
    def to_device(self, device: str):
        """Mover modelo a dispositivo"""
        if self.model:
            self.device = device
            self.model = self.model.to(device)
            logger.info(f"Model moved to {device}")


class FineTuner:
    """Clase para fine-tuning de modelos"""
    
    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None
    
    def prepare_training(
        self,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100
    ):
        """Preparar entrenamiento"""
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # Optimizador
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler (se actualizará con num_training_steps)
        self.warmup_steps = warmup_steps
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        num_training_steps: int = 1000
    ) -> Dict[str, float]:
        """Un paso de entrenamiento"""
        if self.optimizer is None:
            self.prepare_training()
        
        if self.scheduler is None:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps
            )
        
        self.model.train()
        self.model.to(self.device)
        
        # Mover batch a dispositivo
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def save_checkpoint(self, path: str):
        """Guardar checkpoint"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")



