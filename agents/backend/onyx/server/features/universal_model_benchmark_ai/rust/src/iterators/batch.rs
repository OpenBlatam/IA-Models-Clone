//! Batch Iterator
//!
//! Iterator that yields items in batches.

use std::iter::Iterator;

/// Iterator that yields items in batches.
pub struct BatchIterator<I: Iterator> {
    iter: I,
    batch_size: usize,
}

impl<I: Iterator> BatchIterator<I> {
    /// Create a new batch iterator.
    pub fn new(iter: I, batch_size: usize) -> Self {
        Self { iter, batch_size }
    }
    
    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl<I: Iterator> Iterator for BatchIterator<I> {
    type Item = Vec<I::Item>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            if let Some(item) = self.iter.next() {
                batch.push(item);
            } else {
                break;
            }
        }
        
        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let batch_lower = if lower == 0 { 0 } else { (lower - 1) / self.batch_size + 1 };
        let batch_upper = upper.map(|u| if u == 0 { 0 } else { (u - 1) / self.batch_size + 1 });
        (batch_lower, batch_upper)
    }
}

/// Extension trait for iterators to add batch functionality.
pub trait BatchExt: Iterator + Sized {
    /// Group items into batches of specified size.
    fn batches(self, batch_size: usize) -> BatchIterator<Self> {
        BatchIterator::new(self, batch_size)
    }
}

impl<I: Iterator> BatchExt for I {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_iterator() {
        let items = vec![1, 2, 3, 4, 5, 6, 7];
        let batches: Vec<Vec<i32>> = items.into_iter().batches(3).collect();
        assert_eq!(batches, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }
    
    #[test]
    fn test_batch_size_hint() {
        let items = vec![1, 2, 3, 4, 5];
        let mut batches = items.into_iter().batches(2);
        let (lower, upper) = batches.size_hint();
        assert!(lower <= 3);
        assert_eq!(upper, Some(3));
    }
}




