//! Tokenizer Wrapper
//!
//! High-level tokenizer interface with caching and batch support.

use tokenizers::Tokenizer;
use std::sync::Arc;
use std::path::Path;

use super::error::{InferenceError, InferenceResult};

/// Tokenizer wrapper with caching and batch support.
#[derive(Clone)]
pub struct TokenizerWrapper {
    tokenizer: Arc<Tokenizer>,
}

impl TokenizerWrapper {
    /// Load tokenizer from path.
    pub fn from_path(model_path: &Path) -> InferenceResult<Self> {
        let tokenizer_path = model_path.join("tokenizer.json");
        
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| InferenceError::TokenizerError(format!("Failed to load tokenizer from file: {}", e)))?
        } else {
            // Try to load from HuggingFace
            let model_name = model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("gpt2");
            
            Tokenizer::from_pretrained(model_name, None)
                .map_err(|e| InferenceError::TokenizerError(format!("Failed to load tokenizer from HuggingFace: {}", e)))?
        };
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
        })
    }
    
    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> InferenceResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| InferenceError::EncodingError(format!("Failed to encode text: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// Encode batch of texts.
    pub fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> InferenceResult<Vec<Vec<u32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        texts.iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
    
    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> InferenceResult<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }
        
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| InferenceError::DecodingError(format!("Failed to decode tokens: {}", e)))
    }
    
    /// Decode batch of token sequences.
    pub fn decode_batch(&self, token_sequences: &[Vec<u32>], skip_special_tokens: bool) -> InferenceResult<Vec<String>> {
        if token_sequences.is_empty() {
            return Ok(Vec::new());
        }
        
        token_sequences.iter()
            .map(|tokens| self.decode(tokens, skip_special_tokens))
            .collect()
    }
    
    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
    
    /// Get tokenizer reference (for advanced usage).
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }
    
    /// Check if tokenizer is valid.
    pub fn is_valid(&self) -> bool {
        self.vocab_size() > 0
    }
}

