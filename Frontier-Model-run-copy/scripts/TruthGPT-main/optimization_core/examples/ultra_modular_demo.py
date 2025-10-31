"""
Ultra-Modular K/V Cache and Efficient Decoding Demo
Demonstrates prefill and decode phases with K/V cache reuse
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import time

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from modules.attention.ultra_modular_kv_cache import (
    KVCacheModule,
    KVCacheConfig,
    CacheStrategy,
    MemoryLayout,
    create_kv_cache,
    create_kv_cache_config
)

from modules.transformer.ultra_modular_decoder import (
    UltraModularDecoder,
    DecoderConfig,
    DecodePhase,
    MemoryStrategy,
    create_ultra_modular_decoder,
    create_decoder_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraModularDemo:
    """
    Demo for Ultra-Modular K/V Cache and Efficient Decoding.
    
    Demonstrates:
    - Prefill phase: Process entire prompt and populate K/V cache
    - Decode phase: Generate tokens one by one using cached K/V
    - K/V cache reuse for each new token instead of recalculating from scratch
    - Minimized memory overhead and latency between tokens
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = None
        self.demo_results = {}
        
        logger.info(f"Ultra-Modular Demo initialized on {self.device}")
    
    def setup_decoder(self):
        """Setup the ultra-modular decoder."""
        logger.info("Setting up Ultra-Modular Decoder...")
        
        # Create decoder configuration
        decoder_config = create_decoder_config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_sequence_length=4096,
            use_cache=True,
            cache_config=None,  # Will use default
            memory_strategy=MemoryStrategy.BALANCED,
            use_flash_attention=True,
            batch_size=1,
            device=self.device
        )
        
        # Create decoder
        self.decoder = create_ultra_modular_decoder(decoder_config)
        self.decoder.to(self.device)
        
        logger.info("Ultra-Modular Decoder setup complete")
    
    def demo_prefill_phase(self):
        """Demo the prefill phase."""
        logger.info("=== Demo: Prefill Phase ===")
        
        # Create dummy input
        batch_size = 1
        seq_len = 128
        input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(self.device)
        
        start_time = time.time()
        
        # Run prefill phase
        prefill_result = self.decoder.prefill_phase(input_ids)
        
        prefill_time = time.time() - start_time
        
        # Store results
        self.demo_results['prefill_phase'] = {
            'prefill_time': prefill_time,
            'result_time': prefill_result['prefill_time'],
            'seq_len': seq_len,
            'output_shape': prefill_result['output'].shape,
            'cache_state': len(prefill_result['cache_state'])
        }
        
        logger.info(f"Prefill phase completed in {prefill_time:.4f}s")
        logger.info(f"Processed {seq_len} tokens")
        logger.info(f"Output shape: {prefill_result['output'].shape}")
    
    def demo_decode_phase(self):
        """Demo the decode phase."""
        logger.info("=== Demo: Decode Phase ===")
        
        # First, do prefill
        batch_size = 1
        seq_len = 128
        input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(self.device)
        
        prefill_result = self.decoder.prefill_phase(input_ids)
        cache_state = prefill_result['cache_state']
        
        # Now decode single tokens
        decode_times = []
        
        for i in range(10):
            # Get last token
            last_token_ids = torch.randint(0, 50257, (batch_size, 1)).to(self.device)
            
            start_time = time.time()
            
            # Decode phase
            decode_result = self.decoder.decode_phase(last_token_ids, cache_state)
            
            decode_time = time.time() - start_time
            decode_times.append(decode_time)
            
            # Update cache state
            cache_state = decode_result['cache_state']
        
        avg_decode_time = sum(decode_times) / len(decode_times)
        
        # Store results
        self.demo_results['decode_phase'] = {
            'decode_times': decode_times,
            'avg_decode_time': avg_decode_time,
            'total_tokens': len(decode_times)
        }
        
        logger.info(f"Decode phase completed for {len(decode_times)} tokens")
        logger.info(f"Average decode time per token: {avg_decode_time:.4f}s")
    
    def demo_cache_reuse(self):
        """Demo K/V cache reuse."""
        logger.info("=== Demo: K/V Cache Reuse ===")
        
        # Create input
        batch_size = 1
        seq_len = 128
        input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(self.device)
        
        # Prefill phase
        prefill_result = self.decoder.prefill_phase(input_ids)
        cache_state = prefill_result['cache_state']
        
        # Decode phase with cache reuse
        cache_hits = 0
        cache_misses = 0
        
        for i in range(20):
            # Get last token
            last_token_ids = torch.randint(0, 50257, (batch_size, 1)).to(self.device)
            
            # Decode phase
            decode_result = self.decoder.decode_phase(last_token_ids, cache_state)
            
            # Update cache state
            cache_state = decode_result['cache_state']
        
        # Get cache stats
        cache_stats = self.decoder.kv_cache.get_cache_stats()
        
        # Store results
        self.demo_results['cache_reuse'] = {
            'cache_stats': cache_stats,
            'hit_rate': cache_stats['hit_rate'],
            'memory_usage': cache_stats['memory_usage']
        }
        
        logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2f}%")
        logger.info(f"Memory usage: {cache_stats['memory_usage']:.2f} MB")
    
    def demo_full_generation(self):
        """Demo full text generation."""
        logger.info("=== Demo: Full Text Generation ===")
        
        # Create input
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(self.device)
        
        max_length = 50
        temperature = 1.0
        
        start_time = time.time()
        
        # Generate text
        generated_ids = self.decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature
        )
        
        generation_time = time.time() - start_time
        
        # Store results
        self.demo_results['full_generation'] = {
            'generation_time': generation_time,
            'input_length': seq_len,
            'output_length': generated_ids.shape[1],
            'generated_tokens': generated_ids.shape[1] - seq_len
        }
        
        logger.info(f"Full generation completed in {generation_time:.4f}s")
        logger.info(f"Generated {generated_ids.shape[1] - seq_len} tokens")
    
    def demo_performance_comparison(self):
        """Demo performance comparison."""
        logger.info("=== Demo: Performance Comparison ===")
        
        # Test with different configurations
        configs = [
            {
                'name': 'Conservative Memory',
                'memory_strategy': MemoryStrategy.CONSERVATIVE,
                'cache_strategy': CacheStrategy.LRU
            },
            {
                'name': 'Balanced',
                'memory_strategy': MemoryStrategy.BALANCED,
                'cache_strategy': CacheStrategy.ADAPTIVE
            },
            {
                'name': 'Aggressive Speed',
                'memory_strategy': MemoryStrategy.AGGRESSIVE,
                'cache_strategy': CacheStrategy.FIFO
            }
        ]
        
        performance_results = {}
        
        for config in configs:
            logger.info(f"Testing {config['name']}...")
            
            # Create decoder with specific config
            decoder_config = create_decoder_config(
                d_model=512,
                n_heads=8,
                n_layers=6,
                memory_strategy=config['memory_strategy'],
                cache_config=create_kv_cache_config(
                    cache_strategy=config['cache_strategy']
                ),
                device=self.device
            )
            
            test_decoder = create_ultra_modular_decoder(decoder_config)
            test_decoder.to(self.device)
            
            # Run performance test
            input_ids = torch.randint(0, 50257, (1, 128)).to(self.device)
            
            start_time = time.time()
            prefill_result = test_decoder.prefill_phase(input_ids)
            prefill_time = time.time() - start_time
            
            performance_results[config['name']] = {
                'prefill_time': prefill_time,
                'cache_stats': test_decoder.get_performance_stats()
            }
        
        # Store results
        self.demo_results['performance_comparison'] = performance_results
        
        logger.info("Performance comparison completed")
        for name, results in performance_results.items():
            logger.info(f"  {name}: {results['prefill_time']:.4f}s")
    
    def generate_report(self):
        """Generate comprehensive demo report."""
        logger.info("=== Generating Demo Report ===")
        
        report = {
            'device': str(self.device),
            'decoder_config': self.decoder.config.__dict__ if self.decoder else None,
            'results': self.demo_results
        }
        
        # Print summary
        logger.info("Demo Results Summary:")
        for demo_name, results in self.demo_results.items():
            logger.info(f"  {demo_name}: {results}")
        
        return report
    
    def run_complete_demo(self):
        """Run complete demo."""
        logger.info("Starting Ultra-Modular Demo...")
        
        try:
            # Setup
            self.setup_decoder()
            
            # Run demos
            self.demo_prefill_phase()
            self.demo_decode_phase()
            self.demo_cache_reuse()
            self.demo_full_generation()
            self.demo_performance_comparison()
            
            # Generate report
            report = self.generate_report()
            
            logger.info("Ultra-Modular Demo completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise

def main():
    """Main demo function."""
    demo = UltraModularDemo()
    
    try:
        # Run complete demo
        report = demo.run_complete_demo()
        
        logger.info("Demo completed successfully!")
        logger.info(f"Report keys: {report.keys()}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
