//! Safety Module
//!
//! Safety utilities and error handling helpers.

pub mod locks;
pub mod validation;
pub mod context;

// Re-exports
pub use locks::{safe_lock, safe_read_lock, safe_write_lock, safe_unwrap, safe_result};
pub use validation::{
    validate_range,
    validate_positive,
    validate_non_negative,
    validate_non_empty_string,
    validate_non_empty_slice,
    validate_finite,
};
pub use context::ErrorContext;




