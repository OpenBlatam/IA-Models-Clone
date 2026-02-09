//! Configuration Constants
//!
//! Default values and limits for configuration.

/// Default values for configuration.
pub mod defaults {
    pub const MAX_TOKENS: usize = 512;
    pub const TEMPERATURE: f32 = 0.7;
    pub const TOP_P: f32 = 0.9;
    pub const TOP_K: usize = 50;
    pub const BATCH_SIZE: usize = 1;
    pub const REPETITION_PENALTY: f32 = 1.0;
}

/// Limits for configuration values.
pub mod limits {
    pub const MAX_TOKENS_LIMIT: usize = 32768;
    pub const MIN_TOKENS: usize = 1;
    pub const MAX_TEMPERATURE: f32 = 2.0;
    pub const MIN_TEMPERATURE: f32 = 0.0;
    pub const MAX_TOP_P: f32 = 1.0;
    pub const MIN_TOP_P: f32 = 0.0;
    pub const MIN_TOP_K: usize = 1;
    pub const MAX_BATCH_SIZE: usize = 128;
    pub const MIN_BATCH_SIZE: usize = 1;
    pub const MAX_REPETITION_PENALTY: f32 = 2.0;
    pub const MIN_REPETITION_PENALTY: f32 = 1.0;
}




