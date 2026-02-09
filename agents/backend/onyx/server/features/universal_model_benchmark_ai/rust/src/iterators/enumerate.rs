//! Enumerate Iterator
//!
//! Iterator that yields (index, item) pairs starting from a custom index.

use std::iter::Iterator;

/// Iterator that yields (index, item) pairs starting from a custom index.
pub struct EnumerateFrom<I: Iterator> {
    iter: I,
    index: usize,
    start: usize,
}

impl<I: Iterator> EnumerateFrom<I> {
    /// Create a new enumerate iterator starting from a custom index.
    pub fn new(iter: I, start: usize) -> Self {
        Self {
            iter,
            index: start,
            start,
        }
    }
    
    /// Get the starting index.
    pub fn start(&self) -> usize {
        self.start
    }
    
    /// Get the current index.
    pub fn current_index(&self) -> usize {
        self.index
    }
}

impl<I: Iterator> Iterator for EnumerateFrom<I> {
    type Item = (usize, I::Item);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            let idx = self.index;
            self.index += 1;
            (idx, item)
        })
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Extension trait for iterators to add enumerate from functionality.
pub trait EnumerateFromExt: Iterator + Sized {
    /// Enumerate items starting from a custom index.
    fn enumerate_from(self, start: usize) -> EnumerateFrom<Self> {
        EnumerateFrom::new(self, start)
    }
}

impl<I: Iterator> EnumerateFromExt for I {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enumerate_from() {
        let items = vec!['a', 'b', 'c'];
        let enumerated: Vec<(usize, char)> = items.into_iter().enumerate_from(10).collect();
        assert_eq!(enumerated, vec![(10, 'a'), (11, 'b'), (12, 'c')]);
    }
    
    #[test]
    fn test_enumerate_from_start() {
        let items = vec![1, 2, 3];
        let mut enumerated = items.into_iter().enumerate_from(5);
        assert_eq!(enumerated.start(), 5);
        assert_eq!(enumerated.current_index(), 5);
        enumerated.next();
        assert_eq!(enumerated.current_index(), 6);
    }
}




