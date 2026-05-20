"""
Example: Using Specialized Micro-Modules (V5)
Demonstrates the maximum granularity with specialized modules
"""

import torch

# Import from specialized modules
from addiction_recovery_ai.core.layers.micro_modules.normalizers import (
    StandardNormalizer,
    MinMaxNormalizer,
    RobustNormalizer,
    NormalizerFactory
)

from addiction_recovery_ai.core.layers.micro_modules.tokenizers import (
    SimpleTokenizer,
    CharacterTokenizer,
    TokenizerFactory
)

from addiction_recovery_ai.core.layers.micro_modules.padders import (
    ZeroPadder,
    RepeatPadder,
    ReflectPadder,
    PadderFactory
)

from addiction_recovery_ai.core.layers.micro_modules.augmenters import (
    NoiseAugmenter,
    ScaleAugmenter,
    ComposeAugmenter,
    AugmenterFactory
)


# ============================================================================
# Example 1: Specialized Normalizers
# ============================================================================

def example_normalizers():
    """Example of specialized normalizers"""
    print("\n=== Example 1: Specialized Normalizers ===")
    
    data = torch.randn(10) * 5 + 2
    
    # Standard normalization
    std_norm = StandardNormalizer()
    normalized = std_norm.normalize(data)
    print(f"Standard: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Min-Max normalization
    minmax_norm = MinMaxNormalizer()
    normalized = minmax_norm.normalize(data)
    print(f"MinMax: min={normalized.min():.3f}, max={normalized.max():.3f}")
    
    # Robust normalization
    robust_norm = RobustNormalizer()
    normalized = robust_norm.normalize(data)
    print(f"Robust: median={normalized.median():.3f}")
    
    # Using factory
    normalizer = NormalizerFactory.create('standard')
    normalized = normalizer.normalize(data)
    print(f"Factory: {type(normalizer).__name__}")


# ============================================================================
# Example 2: Specialized Tokenizers
# ============================================================================

def example_tokenizers():
    """Example of specialized tokenizers"""
    print("\n=== Example 2: Specialized Tokenizers ===")
    
    text = "Hello world from addiction recovery AI"
    
    # Simple tokenizer
    simple_tokenizer = SimpleTokenizer()
    simple_tokenizer.build_vocab([text])
    tokens = simple_tokenizer.tokenize(text)
    print(f"Simple tokens: {tokens[:5]}...")
    
    # Character tokenizer
    char_tokenizer = CharacterTokenizer()
    tokens = char_tokenizer.tokenize(text)
    print(f"Character tokens length: {len(tokens)}")
    detokenized = char_tokenizer.detokenize(tokens)
    print(f"Detokenized: {detokenized[:20]}...")
    
    # Factory
    tokenizer = TokenizerFactory.create('simple')
    tokenizer.build_vocab([text])
    tokens = tokenizer.tokenize(text)
    print(f"Factory tokenizer: {len(tokens)} tokens")


# ============================================================================
# Example 3: Specialized Padders
# ============================================================================

def example_padders():
    """Example of specialized padders"""
    print("\n=== Example 3: Specialized Padders ===")
    
    sequence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target_length = 10
    
    # Zero padding
    zero_padder = ZeroPadder(pad_value=0)
    padded = zero_padder.pad(sequence, target_length)
    print(f"Zero padded: {padded.tolist()}")
    
    # Repeat padding
    repeat_padder = RepeatPadder()
    padded = repeat_padder.pad(sequence, target_length)
    print(f"Repeat padded: {padded.tolist()}")
    
    # Reflect padding
    reflect_padder = ReflectPadder()
    padded = reflect_padder.pad(sequence, target_length)
    print(f"Reflect padded: {padded.tolist()}")
    
    # Factory
    padder = PadderFactory.create('zero', pad_value=0)
    padded = padder.pad(sequence, target_length)
    print(f"Factory padder: {type(padder).__name__}")


# ============================================================================
# Example 4: Specialized Augmenters
# ============================================================================

def example_augmenters():
    """Example of specialized augmenters"""
    print("\n=== Example 4: Specialized Augmenters ===")
    
    data = torch.randn(10)
    
    # Noise augmentation
    noise_aug = NoiseAugmenter(noise_level=0.1, probability=1.0)
    augmented = noise_aug.augment(data)
    print(f"Noise augmented: mean diff = {(augmented - data).abs().mean():.3f}")
    
    # Scale augmentation
    scale_aug = ScaleAugmenter(scale_range=(0.9, 1.1))
    augmented = scale_aug.augment(data)
    print(f"Scale augmented: mean diff = {(augmented - data).abs().mean():.3f}")
    
    # Compose augmenters
    composed = ComposeAugmenter([
        NoiseAugmenter(noise_level=0.05),
        ScaleAugmenter(scale_range=(0.95, 1.05))
    ])
    augmented = composed.augment(data)
    print(f"Composed augmented: mean diff = {(augmented - data).abs().mean():.3f}")
    
    # Factory
    augmenter = AugmenterFactory.create('noise', noise_level=0.1)
    augmented = augmenter.augment(data)
    print(f"Factory augmenter: {type(augmenter).__name__}")


# ============================================================================
# Example 5: Complete Pipeline with Specialized Modules
# ============================================================================

def example_complete_pipeline():
    """Complete pipeline using specialized modules"""
    print("\n=== Example 5: Complete Pipeline ===")
    
    # 1. Normalize
    normalizer = NormalizerFactory.create('standard')
    data = torch.randn(10) * 5
    data = normalizer.normalize(data)
    print(f"✓ Normalized: mean={data.mean():.3f}")
    
    # 2. Pad
    padder = PadderFactory.create('zero')
    data = padder.pad(data, 20)
    print(f"✓ Padded: shape={data.shape}")
    
    # 3. Augment
    augmenter = AugmenterFactory.create('noise', noise_level=0.1)
    data = augmenter.augment(data)
    print(f"✓ Augmented: shape={data.shape}")
    
    # 4. Tokenize (if text)
    tokenizer = TokenizerFactory.create('simple')
    tokenizer.build_vocab(["hello world"])
    tokens = tokenizer.tokenize("hello world")
    print(f"✓ Tokenized: {len(tokens)} tokens")
    
    print("\nComplete pipeline executed successfully!")


# ============================================================================
# Example 6: Custom Components
# ============================================================================

def example_custom_components():
    """Example of creating custom components"""
    print("\n=== Example 6: Custom Components ===")
    
    # Custom normalizer
    from addiction_recovery_ai.core.layers.micro_modules.normalizers import NormalizerBase
    
    class CustomNormalizer(NormalizerBase):
        def _compute_stats(self, data):
            return {'mean': data.mean().item(), 'scale': 2.0}
        
        def _normalize(self, data, stats):
            return (data - stats['mean']) / stats['scale']
    
    # Register and use
    NormalizerFactory.register('custom', CustomNormalizer)
    custom_norm = NormalizerFactory.create('custom')
    data = torch.randn(10)
    normalized = custom_norm.normalize(data)
    print(f"Custom normalizer: mean={normalized.mean():.3f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Specialized Micro-Modules Examples (V5)")
    print("=" * 60)
    
    try:
        example_normalizers()
        example_tokenizers()
        example_padders()
        example_augmenters()
        example_complete_pipeline()
        example_custom_components()
        
        print("\n" + "=" * 60)
        print("All specialized module examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



