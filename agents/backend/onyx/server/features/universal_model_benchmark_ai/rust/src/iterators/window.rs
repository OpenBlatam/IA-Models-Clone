//! Window Iterator
//!
//! Iterator that yields sliding windows of items.

use std::iter::Iterator;

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
    
    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_window_iterator() {
        let items = vec![1, 2, 3, 4, 5];
        let windows: Vec<Vec<i32>> = items.into_iter().windows(3).collect();
        assert_eq!(windows, vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]);
    }
    
    #[test]
    fn test_window_size() {
        let items = vec![1, 2, 3];
        let mut windows = items.into_iter().windows(2);
        assert_eq!(windows.window_size(), 2);
    }
}




