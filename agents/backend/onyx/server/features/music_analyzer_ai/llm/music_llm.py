"""
LLM Integration for Music Analysis
Use LLMs for text-based music analysis and generation
Implements LoRA, P-tuning, and advanced fine-tuning techniques
"""

from typing import Dict, Any, Optional, List, Union
import logging
import json
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available for LLM features")

# Try to import PEFT for LoRA and P-tuning
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        PromptTuningConfig,
        PromptTuningInit,
        get_peft_model as get_peft_model_pt
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available, LoRA and P-tuning features disabled")

# Try to import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("BitsAndBytes not available, quantization disabled")


class MusicLLMAnalyzer:
    """
    LLM-based music analysis with LoRA, P-tuning, and quantization support
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        use_lora: bool = False,
        use_quantization: bool = False,
        lora_config: Optional[Dict] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required for LLM features")
        
        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora and PEFT_AVAILABLE
        self.use_quantization = use_quantization and BITSANDBYTES_AVAILABLE
        self.lora_config = lora_config or {}
        self.load_in_8bit = load_in_8bit and BITSANDBYTES_AVAILABLE
        self.load_in_4bit = load_in_4bit and BITSANDBYTES_AVAILABLE
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optional quantization
            model_kwargs = {}
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            # Try to determine model type
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
            except:
                # Fallback to seq2seq models
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    **model_kwargs,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
            
            if device != "cuda" or "device_map" not in model_kwargs:
                self.model = self.model.to(device)
            
            # Apply LoRA if requested
            if self.use_lora:
                self._apply_lora()
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" and not self.load_in_8bit and not self.load_in_4bit else -1
            )
            
            logger.info(f"Loaded LLM model {model_name} on {device} "
                       f"(LoRA: {self.use_lora}, Quantization: {self.load_in_8bit or self.load_in_4bit})")
        
        except Exception as e:
            logger.error(f"Could not load LLM model {model_name}: {str(e)}", exc_info=True)
            self.generator = None
            self.model = None
    
    def _apply_lora(self):
        """Apply LoRA to the model for efficient fine-tuning"""
        if not PEFT_AVAILABLE or self.model is None:
            return
        
        try:
            # Determine target modules based on model architecture
            if "gpt" in self.model_name.lower() or "llama" in self.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
            elif "t5" in self.model_name.lower():
                target_modules = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
            else:
                target_modules = self.lora_config.get("target_modules", ["q_proj", "v_proj"])
            
            lora_config = LoraConfig(
                r=self.lora_config.get("r", 8),
                lora_alpha=self.lora_config.get("lora_alpha", 16),
                target_modules=target_modules,
                lora_dropout=self.lora_config.get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"Applied LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        except Exception as e:
            logger.error(f"Error applying LoRA: {str(e)}", exc_info=True)
    
    def analyze_music_description(
        self,
        description: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        use_mixed_precision: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze music based on text description with better generation parameters
        
        Args:
            description: Music description to analyze
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            use_mixed_precision: Use mixed precision for faster inference
        
        Returns:
            Analysis results
        """
        if self.generator is None:
            return {"error": "LLM model not available"}
        
        prompt = f"Analyze this music description: {description}\nAnalysis:"
        
        try:
            # Use autocast for mixed precision if available
            with torch.cuda.amp.autocast(enabled=(use_mixed_precision and self.device == "cuda")):
                generated = self.generator(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Reduce repetition
                )
            
            analysis_text = generated[0]["generated_text"]
            # Remove prompt from output
            if analysis_text.startswith(prompt):
                analysis_text = analysis_text[len(prompt):].strip()
            
            return {
                "description": description,
                "analysis": analysis_text,
                "model": self.model_name,
                "generation_params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            }
        except Exception as e:
            logger.error(f"LLM analysis error: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def generate_music_recommendations(
        self,
        user_preferences: str,
        num_recommendations: int = 5
    ) -> Dict[str, Any]:
        """Generate music recommendations based on text"""
        if self.generator is None:
            return {"error": "LLM model not available"}
        
        prompt = f"Based on these preferences: {user_preferences}\nRecommend {num_recommendations} songs:\n"
        
        try:
            generated = self.generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.8
            )
            
            recommendations_text = generated[0]["generated_text"]
            
            return {
                "preferences": user_preferences,
                "recommendations": recommendations_text,
                "num_recommendations": num_recommendations
            }
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
            return {"error": str(e)}
    
    def summarize_analysis(
        self,
        analysis_data: Dict[str, Any],
        max_length: int = 150,
        temperature: float = 0.5
    ) -> str:
        """Summarize analysis results using LLM with better prompting"""
        if self.generator is None:
            return "LLM model not available"
        
        # Convert analysis to text
        analysis_text = json.dumps(analysis_data, indent=2)
        
        prompt = f"Summarize this music analysis in 2-3 sentences:\n{analysis_text}\nSummary:"
        
        try:
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                generated = self.generator(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            summary = generated[0]["generated_text"]
            if summary.startswith(prompt):
                summary = summary[len(prompt):].strip()
            
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    def fine_tune_with_lora(
        self,
        train_dataset,
        output_dir: str = "./lora_model",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        lora_r: int = 8,
        lora_alpha: int = 16
    ):
        """
        Fine-tune the model using LoRA
        
        Args:
            train_dataset: Training dataset
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        if not PEFT_AVAILABLE:
            logger.error("PEFT required for LoRA fine-tuning")
            return
        
        if self.model is None:
            logger.error("Model not loaded")
            return
        
        try:
            # Apply LoRA if not already applied
            if not self.use_lora:
                self.lora_config = {"r": lora_r, "lora_alpha": lora_alpha}
                self.use_lora = True
                self._apply_lora()
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.device == "cuda",  # Use mixed precision
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                warmup_steps=100,
                weight_decay=0.01,
                optim="adamw_torch"
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator
            )
            
            # Train
            trainer.train()
            
            # Save model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"LoRA fine-tuning completed, model saved to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error in LoRA fine-tuning: {str(e)}", exc_info=True)


class MusicTextEmbedder:
    """
    Text embedding for music-related text
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers required")
        
        self.use_sentence_transformers = False
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_sentence_transformers = True
        except ImportError:
            try:
                # Fallback to transformers
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                logger.info("Using transformers fallback for embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {str(e)}")
                self.model = None
                self.tokenizer = None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Get text embedding"""
        if self.model is None:
            return None
        
        try:
            if self.use_sentence_transformers:
                # Use sentence-transformers
                embedding = self.model.encode(text)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            else:
                # Use transformers model
                import torch
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return None
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        if self.model is None:
            return 0.0
        
        try:
            emb1 = self.embed_text(text1)
            emb2 = self.embed_text(text2)
            
            if emb1 is None or emb2 is None:
                return 0.0
            
            # Cosine similarity
            import numpy as np
            emb1_arr = np.array(emb1)
            emb2_arr = np.array(emb2)
            similarity = np.dot(emb1_arr, emb2_arr) / (np.linalg.norm(emb1_arr) * np.linalg.norm(emb2_arr))
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return 0.0

