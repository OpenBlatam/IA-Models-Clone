//! Sampling Strategies
//!
//! Token sampling strategies for text generation.

use super::config::{SamplingConfig, SamplingStrategy};
use rand::Rng;

/// Sample tokens using configured strategy.
pub fn sample_token(
    logits: &[f32],
    config: &SamplingConfig,
) -> usize {
    match config.strategy {
        SamplingStrategy::Greedy => greedy_sample(logits),
        SamplingStrategy::TopK => top_k_sample(logits, 50),
        SamplingStrategy::TopP => top_p_sample(logits, 0.9),
        SamplingStrategy::Temperature => temperature_sample(logits, 0.7),
        SamplingStrategy::Nucleus => nucleus_sample(logits, 0.9),
    }
}

/// Greedy sampling (argmax).
fn greedy_sample(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Top-K sampling.
fn top_k_sample(logits: &[f32], k: usize) -> usize {
    let mut indexed: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    
    // Sample from top-k
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..indexed.len().min(k));
    indexed[idx].0
}

/// Top-P (nucleus) sampling.
fn top_p_sample(logits: &[f32], p: f32) -> usize {
    nucleus_sample(logits, p)
}

/// Temperature sampling.
fn temperature_sample(logits: &[f32], temperature: f32) -> usize {
    let scaled: Vec<f32> = logits.iter()
        .map(|&x| x / temperature)
        .collect();
    
    // Apply softmax and sample
    let exp_sum: f32 = scaled.iter().map(|&x| x.exp()).sum();
    let probs: Vec<f32> = scaled.iter().map(|&x| x.exp() / exp_sum).collect();
    
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if r <= cumsum {
            return i;
        }
    }
    
    probs.len() - 1
}

/// Nucleus sampling (top-p).
fn nucleus_sample(logits: &[f32], p: f32) -> usize {
    let mut indexed: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    
    // Calculate cumulative probability
    let exp_sum: f32 = indexed.iter().map(|(_, v)| v.exp()).sum();
    let mut cumsum = 0.0;
    let mut cutoff = indexed.len();
    
    for (i, (_, v)) in indexed.iter().enumerate() {
        cumsum += v.exp() / exp_sum;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }
    
    // Sample from nucleus
    let mut rng = rand::thread_rng();
    let idx = rng.gen_range(0..cutoff);
    indexed[idx].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::config::SamplingConfig;
    
    #[test]
    fn test_greedy_sample() {
        let logits = vec![0.1, 0.5, 0.2, 0.3];
        let config = SamplingConfig {
            strategy: SamplingStrategy::Greedy,
            seed: None,
        };
        let token = sample_token(&logits, &config);
        assert_eq!(token, 1); // Index of max value
    }
}

