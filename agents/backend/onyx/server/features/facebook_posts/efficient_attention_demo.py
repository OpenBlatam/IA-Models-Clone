"""
Demonstration of Efficient Fine-tuning with Advanced Attention Mechanisms.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from efficient_finetuning_system import (
    LoRAConfig, PtuningConfig, AdapterConfig,
    EfficientFineTuner, EfficientTrainer,
    create_efficient_finetuner
)
from attention_mechanisms import (
    AttentionConfig, MultiHeadAttention, RotaryPositionalEmbedding,
    TransformerBlock, create_attention_mask
)
from typing import Dict, Any, List


class AdvancedTransformerWithEfficiencyDemo:
    """Demo class combining efficient fine-tuning with advanced attention."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
    
    def demo_lora_with_attention(self):
        """Demonstrate LoRA fine-tuning with custom attention."""
        print("\n" + "="*60)
        print("🔧 DEMO: LoRA Fine-tuning with Advanced Attention")
        print("="*60)
        
        # Load base model
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create model with LoRA
        model, fine_tuner = create_efficient_finetuner(
            model_name=model_name,
            method="lora",
            rank=16,
            alpha=32.0,
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
            dropout=0.1
        )
        
        model = model.to(self.device)
        
        # Test generation before fine-tuning
        prompt = "The future of artificial intelligence"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        print(f"\n📝 Original Model Generation:")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   {generated_text}")
        
        # Show LoRA statistics
        trainable_params = len(fine_tuner.get_trainable_parameters())
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n📊 LoRA Statistics:")
        print(f"   Rank: {fine_tuner.config.rank}")
        print(f"   Alpha: {fine_tuner.config.alpha}")
        print(f"   Target modules: {fine_tuner.config.target_modules}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Efficiency: {trainable_params/total_params*100:.2f}% trainable")
        
        return model, fine_tuner, tokenizer
    
    def demo_ptuning_with_attention(self):
        """Demonstrate P-tuning v2 with custom attention."""
        print("\n" + "="*60)
        print("🎯 DEMO: P-tuning v2 with Advanced Attention")
        print("="*60)
        
        # Create P-tuning configuration
        ptuning_config = PtuningConfig(
            num_virtual_tokens=20,
            token_dim=768,
            num_transformer_submodules=2,
            num_attention_heads=12,
            num_layers=12,
            encoder_hidden_size=768,
            prefix_projection=True
        )
        
        # Load base model
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Apply P-tuning
        fine_tuner = EfficientFineTuner(model, method="ptuning", config=ptuning_config)
        
        # Test prefix encoder
        batch_size = 2
        prefix_states = fine_tuner.prefix_encoder(batch_size)
        
        print(f"\n🎯 P-tuning v2 Statistics:")
        print(f"   Virtual tokens: {ptuning_config.num_virtual_tokens}")
        print(f"   Token dimension: {ptuning_config.token_dim}")
        print(f"   Prefix projection: {ptuning_config.prefix_projection}")
        print(f"   Prefix states shape: {prefix_states.shape}")
        
        # Show parameter efficiency
        trainable_params = len(fine_tuner.get_trainable_parameters())
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Efficiency: {trainable_params/total_params*100:.2f}% trainable")
        
        return model, fine_tuner
    
    def demo_custom_attention_mechanisms(self):
        """Demonstrate custom attention mechanisms."""
        print("\n" + "="*60)
        print("🧠 DEMO: Custom Attention Mechanisms & Positional Encodings")
        print("="*60)
        
        # Attention configuration
        attention_config = AttentionConfig(
            hidden_size=768,
            num_heads=12,
            dropout_rate=0.1,
            use_rotary=True,
            use_relative_position=True,
            use_flash_attention=True,
            max_position_embeddings=2048
        )
        
        # Create attention components
        attention_layer = MultiHeadAttention(attention_config)
        transformer_block = TransformerBlock(attention_config)
        rotary_emb = RotaryPositionalEmbedding(64)  # head_dim
        
        # Move to device
        attention_layer = attention_layer.to(self.device)
        transformer_block = transformer_block.to(self.device)
        rotary_emb = rotary_emb.to(self.device)
        
        # Test with sample data
        batch_size, seq_len = 2, 128
        hidden_states = torch.randn(batch_size, seq_len, attention_config.hidden_size).to(self.device)
        
        # Create attention mask
        attention_mask = create_attention_mask(
            torch.ones(batch_size, seq_len),
            mask_type="causal"
        ).to(self.device)
        
        print(f"\n🧠 Attention Configuration:")
        print(f"   Hidden size: {attention_config.hidden_size}")
        print(f"   Number of heads: {attention_config.num_heads}")
        print(f"   Head dimension: {attention_config.hidden_size // attention_config.num_heads}")
        print(f"   Using RoPE: {attention_config.use_rotary}")
        print(f"   Using Flash Attention: {attention_config.use_flash_attention}")
        print(f"   Using Relative Position: {attention_config.use_relative_position}")
        
        # Test attention layer
        print(f"\n🔍 Testing Attention Mechanisms:")
        with torch.no_grad():
            # Multi-head attention
            attention_outputs = attention_layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            attention_output = attention_outputs[0]
            attention_weights = attention_outputs[1] if len(attention_outputs) > 1 else None
            
            print(f"   Input shape: {hidden_states.shape}")
            print(f"   Attention output shape: {attention_output.shape}")
            print(f"   Attention weights shape: {attention_weights.shape if attention_weights is not None else 'None'}")
            
            # Transformer block
            block_outputs = transformer_block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            block_output = block_outputs[0]
            print(f"   Transformer block output shape: {block_output.shape}")
            
            # Test rotary embeddings
            query = torch.randn(batch_size, attention_config.num_heads, seq_len, 64).to(self.device)
            key = torch.randn(batch_size, attention_config.num_heads, seq_len, 64).to(self.device)
            
            q_rot, k_rot = rotary_emb(query, key, seq_len)
            print(f"   Rotary embedding - Query shape: {q_rot.shape}")
            print(f"   Rotary embedding - Key shape: {k_rot.shape}")
        
        return attention_layer, transformer_block
    
    def demo_efficiency_comparison(self):
        """Compare efficiency of different fine-tuning methods."""
        print("\n" + "="*60)
        print("📊 DEMO: Efficiency Comparison of Fine-tuning Methods")
        print("="*60)
        
        model_name = "gpt2"
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        total_params = sum(p.numel() for p in base_model.parameters())
        
        methods = [
            ("LoRA (r=16)", "lora", {"rank": 16, "alpha": 32.0}),
            ("LoRA (r=8)", "lora", {"rank": 8, "alpha": 16.0}),
            ("P-tuning v2", "ptuning", {"num_virtual_tokens": 20}),
            ("Adapter", "adapter", {"reduction_factor": 16}),
            ("BitFit", "bitfit", {})
        ]
        
        print(f"\n📊 Fine-tuning Method Comparison:")
        print(f"{'Method':<15} {'Trainable':<12} {'Total':<12} {'Efficiency':<12}")
        print("-" * 55)
        
        for method_name, method_type, kwargs in methods:
            # Create fresh model for each method
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            try:
                if method_type == "lora":
                    config = LoRAConfig(**kwargs)
                elif method_type == "ptuning":
                    config = PtuningConfig(**kwargs)
                elif method_type == "adapter":
                    config = AdapterConfig(**kwargs)
                else:
                    config = None
                
                fine_tuner = EfficientFineTuner(model, method_type, config)
                trainable_params = len(fine_tuner.get_trainable_parameters())
                efficiency = trainable_params / total_params * 100
                
                print(f"{method_name:<15} {trainable_params:<12,} {total_params:<12,} {efficiency:<12.2f}%")
                
            except Exception as e:
                print(f"{method_name:<15} {'Error':<12} {total_params:<12,} {'N/A':<12}")
    
    def demo_training_workflow(self):
        """Demonstrate complete training workflow."""
        print("\n" + "="*60)
        print("🚀 DEMO: Complete Efficient Fine-tuning Workflow")
        print("="*60)
        
        # Create model with LoRA
        model, fine_tuner, tokenizer = self.demo_lora_with_attention()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./efficient_training_demo",
            learning_rate=2e-4,
            num_train_epochs=1,  # Short demo
            per_device_train_batch_size=2,
            save_steps=1000,
            logging_steps=100,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None
        )
        
        # Create trainer
        trainer = EfficientTrainer(model, fine_tuner, training_args)
        
        print(f"\n🚀 Training Configuration:")
        print(f"   Method: {fine_tuner.method}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Mixed precision: {training_args.fp16}")
        print(f"   Trainable parameters: {len(fine_tuner.get_trainable_parameters()):,}")
        
        # Save efficient weights (demo)
        print(f"\n💾 Saving efficient weights...")
        fine_tuner.save_efficient_weights("demo_lora_weights.pt")
        
        # Load efficient weights (demo)
        print(f"📁 Loading efficient weights...")
        fine_tuner.load_efficient_weights("demo_lora_weights.pt")
        
        print(f"✅ Training workflow demonstration complete!")
    
    def run_all_demos(self):
        """Run all demonstration functions."""
        print("🎯 Starting Efficient Fine-tuning & Attention Mechanisms Demo")
        print("=" * 80)
        
        try:
            # Demo 1: LoRA with attention
            self.demo_lora_with_attention()
            
            # Demo 2: P-tuning with attention
            self.demo_ptuning_with_attention()
            
            # Demo 3: Custom attention mechanisms
            self.demo_custom_attention_mechanisms()
            
            # Demo 4: Efficiency comparison
            self.demo_efficiency_comparison()
            
            # Demo 5: Training workflow
            self.demo_training_workflow()
            
            print("\n" + "="*80)
            print("🎉 All demos completed successfully!")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            raise


def main():
    """Main function to run the demonstration."""
    demo = AdvancedTransformerWithEfficiencyDemo()
    demo.run_all_demos()


if __name__ == "__main__":
    main()






