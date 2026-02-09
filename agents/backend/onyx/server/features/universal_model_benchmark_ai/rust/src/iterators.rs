//! Iterator Adapters - Useful iterator extensions.
//!
//! Provides:
//! - Custom iterator adapters
//! - Batch processing iterators
//! - Windowed iterators
//! - Chunked iterators

use std::iter::Iterator;

// ════════════════════════════════════════════════════════════════════════════════
// BATCH ITERATOR
// ════════════════════════════════════════════════════════════════════════════════

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
}

/// Extension trait for iterators to add batch functionality.
pub trait BatchExt: Iterator + Sized {
    /// Group items into batches of specified size.
    fn batches(self, batch_size: usize) -> BatchIterator<Self> {
        BatchIterator::new(self, batch_size)
    }
}

impl<I: Iterator> BatchExt for I {}

// ════════════════════════════════════════════════════════════════════════════════
// WINDOW ITERATOR
// ════════════════════════════════════════════════════════════════════════════════

/// Iterator that yields sliding windows of items.
pub struct WindowIterator<I: Iterator> {
    iter: I,
    window_size: usize,
    buffer: Vec<I::Item>,
}

impl<I: Iterator> WindowIterator<I>
where
    I::Item: Clone,
{
    /// Create a new window iterator.
    pub fn new(iter: I, window_size: usize) -> Self {
        Self {
            iter,
            window_size,
            buffer: Vec::with_capacity(window_size),
        }
    }
}

impl<I: Iterator> Iterator for WindowIterator<I>
where
    I::Item: Clone,
{
    type Item = Vec<I::Item>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Fill buffer if needed
        while self.buffer.len() < self.window_size {
            if let Some(item) = self.iter.next() {
                self.buffer.push(item);
            } else {
                break;
            }
        }
        
        if self.buffer.len() < self.window_size {
            return None;
        }
        
        // Return window and advance
        let window = self.buffer.clone();
        self.buffer.remove(0);
        
        Some(window)
    }
}

/// Extension trait for iterators to add window functionality.
pub trait WindowExt: Iterator + Sized {
    /// Create sliding windows of specified size.
    fn windows(self, window_size: usize) -> WindowIterator<Self>
    where
        Self::Item: Clone,
    {
        WindowIterator::new(self, window_size)
    }
}

impl<I: Iterator> WindowExt for I {}

// ════════════════════════════════════════════════════════════════════════════════
// ENUMERATE WITH INDEX
// ════════════════════════════════════════════════════════════════════════════════

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
}

/// Extension trait for iterators to add enumerate from functionality.
pub trait EnumerateFromExt: Iterator + Sized {
    /// Enumerate items starting from a custom index.
    fn enumerate_from(self, start: usize) -> EnumerateFrom<Self> {
        EnumerateFrom::new(self, start)
    }
}

impl<I: Iterator> EnumerateFromExt for I {}

// ════════════════════════════════════════════════════════════════════════════════
// TAKE WHILE INCLUSIVE
// ════════════════════════════════════════════════════════════════════════════════

/// Iterator that takes items while predicate is true, including the first false item.
pub struct TakeWhileInclusive<I: Iterator, P> {
    iter: I,
    predicate: P,
    done: bool,
}

impl<I: Iterator, P: FnMut(&I::Item) -> bool> TakeWhileInclusive<I, P> {
    /// Create a new take while inclusive iterator.
    pub fn new(iter: I, predicate: P) -> Self {
        Self {
            iter,
            predicate,
            done: false,
        }
    }
}

impl<I: Iterator, P: FnMut(&I::Item) -> bool> Iterator for TakeWhileInclusive<I, P> {
    type Item = I::Item;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        
        if let Some(item) = self.iter.next() {
            if (self.predicate)(&item) {
                Some(item)
            } else {
                self.done = true;
                Some(item)  // Include the first false item
            }
        } else {
            self.done = true;
            None
        }
    }
}

/// Extension trait for iterators to add take while inclusive functionality.
pub trait TakeWhileInclusiveExt: Iterator + Sized {
    /// Take items while predicate is true, including the first false item.
    fn take_while_inclusive<P>(self, predicate: P) -> TakeWhileInclusive<Self, P>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        TakeWhileInclusive::new(self, predicate)
    }
}

impl<I: Iterator> TakeWhileInclusiveExt for I {}

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
    fn test_window_iterator() {
        let items = vec![1, 2, 3, 4, 5];
        let windows: Vec<Vec<i32>> = items.into_iter().windows(3).collect();
        assert_eq!(windows, vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]);
    }
    
    #[test]
    fn test_enumerate_from() {
        let items = vec!['a', 'b', 'c'];
        let enumerated: Vec<(usize, char)> = items.into_iter().enumerate_from(10).collect();
        assert_eq!(enumerated, vec![(10, 'a'), (11, 'b'), (12, 'c')]);
    }
    
    #[test]
    fn test_take_while_inclusive() {
        let items = vec![1, 2, 3, 4, 5, 6];
        let taken: Vec<i32> = items.into_iter()
            .take_while_inclusive(|&x| x < 4)
            .collect();
        assert_eq!(taken, vec![1, 2, 3, 4]);  // Includes 4
    }
}












