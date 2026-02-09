/*
 * Data Processor - High-performance data processing
 * 
 * Refactored with:
 * - Better batch processing
 * - Tokenization support
 * - Prompt templating
 * - Data validation
 * - Error handling
 */

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{Result, BenchmarkError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessorConfig {
    pub batch_size: usize,
    pub max_length: Option<usize>,
    pub padding: bool,
    pub truncation: bool,
}

impl Default for DataProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_length: Some(512),
            padding: true,
            truncation: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataProcessor {
    config: DataProcessorConfig,
}

impl DataProcessor {
    /// Create a new data processor.
    pub fn new(config: Option<DataProcessorConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }
    
    /// Process a batch of strings.
    pub fn process_batch(&self, data: &[String]) -> Result<Vec<Vec<u32>>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut results = Vec::with_capacity(data.len());
        
        for text in data {
            let processed = self.process_text(text)?;
            results.push(processed);
        }
        
        // Apply padding if needed
        if self.config.padding {
            self.pad_batch(&mut results)?;
        }
        
        Ok(results)
    }
    
    /// Process a single text string.
    fn process_text(&self, text: &str) -> Result<Vec<u32>> {
        let mut bytes: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|&b| b as u32)
            .collect();
        
        // Apply truncation if needed
        if let Some(max_len) = self.config.max_length {
            if self.config.truncation && bytes.len() > max_len {
                bytes.truncate(max_len);
            }
        }
        
        Ok(bytes)
    }
    
    /// Pad batch to same length.
    fn pad_batch(&self, batch: &mut [Vec<u32>]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        
        let max_len = batch.iter()
            .map(|v| v.len())
            .max()
            .ok_or_else(|| BenchmarkError::invalid_input("Empty batch"))?;
        
        for item in batch.iter_mut() {
            let pad_len = max_len - item.len();
            if pad_len > 0 {
                item.extend(vec![0u32; pad_len]);
            }
        }
        
        Ok(())
    }
    
    /// Format prompt with template variables.
    pub fn format_prompt(
        &self,
        template: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let mut result = template.to_string();
        
        for (key, value) in variables {
            let placeholder = format!("{{{}}}", key);
            if !result.contains(&placeholder) {
                return Err(BenchmarkError::invalid_input(
                    format!("Template variable '{}' not found in template", key)
                ));
            }
            result = result.replace(&placeholder, value);
        }
        
        // Check for unsubstituted variables
        if result.contains('{') && result.contains('}') {
            return Err(BenchmarkError::invalid_input(
                "Unsubstituted template variables found"
            ));
        }
        
        Ok(result)
    }
    
    /// Format batch of prompts.
    pub fn format_prompt_batch(
        &self,
        template: &str,
        variables_batch: &[HashMap<String, String>],
    ) -> Result<Vec<String>> {
        variables_batch
            .iter()
            .map(|vars| self.format_prompt(template, vars))
            .collect()
    }
    
    /// Validate data batch.
    pub fn validate_batch(&self, data: &[String]) -> Result<()> {
        if data.is_empty() {
            return Err(BenchmarkError::invalid_input("Empty batch"));
        }
        
        for (i, text) in data.iter().enumerate() {
            if text.is_empty() {
                return Err(BenchmarkError::invalid_input(
                    format!("Empty text at index {}", i)
                ));
            }
            
            if let Some(max_len) = self.config.max_length {
                if text.len() > max_len && !self.config.truncation {
                    return Err(BenchmarkError::invalid_input(
                        format!("Text at index {} exceeds max_length {}", i, max_len)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get configuration.
    pub fn config(&self) -> &DataProcessorConfig {
        &self.config
    }
    
    /// Update configuration.
    pub fn set_config(&mut self, config: DataProcessorConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_batch() {
        let processor = DataProcessor::new(None);
        let data = vec!["hello".to_string(), "world".to_string()];
        let result = processor.process_batch(&data).unwrap();
        assert_eq!(result.len(), 2);
    }
    
    #[test]
    fn test_format_prompt() {
        let processor = DataProcessor::new(None);
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        
        let template = "Hello, {name}!";
        let result = processor.format_prompt(template, &vars).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }
    
    #[test]
    fn test_pad_batch() {
        let processor = DataProcessor::new(None);
        let mut batch = vec![
            vec![1, 2, 3],
            vec![4, 5],
        ];
        processor.pad_batch(&mut batch).unwrap();
        assert_eq!(batch[0].len(), 3);
        assert_eq!(batch[1].len(), 3);
        assert_eq!(batch[1][2], 0);
    }
}
