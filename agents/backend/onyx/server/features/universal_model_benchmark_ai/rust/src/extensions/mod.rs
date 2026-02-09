//! Extensions Module
//!
//! Extension traits for common types.

pub mod f64_slice;
pub mod result;
pub mod string;
pub mod vec;
pub mod option;

// Re-exports
pub use f64_slice::F64SliceExt;
pub use result::ResultExt;
pub use string::StringExt;
pub use vec::VecExt;
pub use option::OptionExt;




