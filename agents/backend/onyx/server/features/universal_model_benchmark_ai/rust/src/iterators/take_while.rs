//! Take While Inclusive Iterator
//!
//! Iterator that takes items while predicate is true, including the first false item.

use std::iter::Iterator;

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
    fn test_take_while_inclusive() {
        let items = vec![1, 2, 3, 4, 5, 6];
        let taken: Vec<i32> = items.into_iter()
            .take_while_inclusive(|&x| x < 4)
            .collect();
        assert_eq!(taken, vec![1, 2, 3, 4]);  // Includes 4
    }
    
    #[test]
    fn test_take_while_inclusive_all() {
        let items = vec![1, 2, 3];
        let taken: Vec<i32> = items.into_iter()
            .take_while_inclusive(|&x| x < 10)
            .collect();
        assert_eq!(taken, vec![1, 2, 3]);
    }
}




