//! Iterators Module
//!
//! Custom iterator adapters and extensions.

pub mod batch;
pub mod window;
pub mod enumerate;
pub mod take_while;

// Re-exports
pub use batch::{BatchIterator, BatchExt};
pub use window::{WindowIterator, WindowExt};
pub use enumerate::{EnumerateFrom, EnumerateFromExt};
pub use take_while::{TakeWhileInclusive, TakeWhileInclusiveExt};




