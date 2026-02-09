//! Distributed Locking
//!
//! Provides distributed locking mechanisms.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Lock state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockState {
    Unlocked,
    Locked,
    Expired,
}

/// Distributed lock
#[pyclass]
pub struct DistributedLock {
    lock_id: String,
    owner: Option<String>,
    acquired_at: Option<Instant>,
    ttl_ms: u64,
    state: Arc<Mutex<LockState>>,
}

#[pymethods]
impl DistributedLock {
    #[new]
    pub fn new(lock_id: String, ttl_ms: u64) -> Self {
        Self {
            lock_id,
            owner: None,
            acquired_at: None,
            ttl_ms,
            state: Arc::new(Mutex::new(LockState::Unlocked)),
        }
    }

    pub fn acquire(&mut self, owner: String) -> PyResult<bool> {
        let mut state = self.state.lock().unwrap();
        
        match *state {
            LockState::Unlocked | LockState::Expired => {
                *state = LockState::Locked;
                self.owner = Some(owner);
                self.acquired_at = Some(Instant::now());
                Ok(true)
            }
            LockState::Locked => {
                // Check if expired
                if let Some(acquired_at) = self.acquired_at {
                    if acquired_at.elapsed().as_millis() as u64 > self.ttl_ms {
                        *state = LockState::Locked;
                        self.owner = Some(owner);
                        self.acquired_at = Some(Instant::now());
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
        }
    }

    pub fn release(&mut self, owner: String) -> PyResult<bool> {
        let mut state = self.state.lock().unwrap();
        
        if let Some(ref lock_owner) = self.owner {
            if lock_owner == &owner && *state == LockState::Locked {
                *state = LockState::Unlocked;
                self.owner = None;
                self.acquired_at = None;
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    pub fn is_locked(&self) -> bool {
        let state = self.state.lock().unwrap();
        *state == LockState::Locked && self.is_valid()
    }

    pub fn get_owner(&self) -> Option<String> {
        self.owner.clone()
    }

    pub fn get_remaining_ttl_ms(&self) -> Option<u64> {
        if let Some(acquired_at) = self.acquired_at {
            let elapsed = acquired_at.elapsed().as_millis() as u64;
            if elapsed < self.ttl_ms {
                Some(self.ttl_ms - elapsed)
            } else {
                Some(0)
            }
        } else {
            None
        }
    }

    fn is_valid(&self) -> bool {
        if let Some(acquired_at) = self.acquired_at {
            acquired_at.elapsed().as_millis() as u64 < self.ttl_ms
        } else {
            false
        }
    }
}

/// Lock manager
#[pyclass]
pub struct LockManager {
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<DistributedLock>>>>>,
    default_ttl_ms: u64,
}

#[pymethods]
impl LockManager {
    #[new]
    pub fn new(default_ttl_ms: u64) -> Self {
        Self {
            locks: Arc::new(Mutex::new(HashMap::new())),
            default_ttl_ms,
        }
    }

    pub fn acquire_lock(&self, lock_id: String, owner: String, ttl_ms: Option<u64>) -> PyResult<bool> {
        let mut locks = self.locks.lock().unwrap();
        let ttl = ttl_ms.unwrap_or(self.default_ttl_ms);
        
        let lock = locks
            .entry(lock_id.clone())
            .or_insert_with(|| Arc::new(Mutex::new(DistributedLock::new(lock_id, ttl))))
            .clone();
        
        let mut lock_guard = lock.lock().unwrap();
        lock_guard.acquire(owner)
    }

    pub fn release_lock(&self, lock_id: String, owner: String) -> PyResult<bool> {
        let locks = self.locks.lock().unwrap();
        if let Some(lock) = locks.get(&lock_id) {
            let mut lock_guard = lock.lock().unwrap();
            lock_guard.release(owner)
        } else {
            Ok(false)
        }
    }

    pub fn is_locked(&self, lock_id: String) -> bool {
        let locks = self.locks.lock().unwrap();
        if let Some(lock) = locks.get(&lock_id) {
            let lock_guard = lock.lock().unwrap();
            lock_guard.is_locked()
        } else {
            false
        }
    }

    pub fn cleanup_expired(&self) -> usize {
        let mut locks = self.locks.lock().unwrap();
        let mut removed = 0;
        
        locks.retain(|_, lock| {
            let lock_guard = lock.lock().unwrap();
            if !lock_guard.is_locked() {
                removed += 1;
                false
            } else {
                true
            }
        });
        
        removed
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let locks = self.locks.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total_locks", locks.len())?;
            dict.set_item("locked_count", locks.values()
                .filter(|lock| {
                    let lock_guard = lock.lock().unwrap();
                    lock_guard.is_locked()
                })
                .count())?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
pub fn create_lock_manager(default_ttl_ms: Option<u64>) -> LockManager {
    LockManager::new(default_ttl_ms.unwrap_or(5000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_lock() {
        let mut lock = DistributedLock::new("test-lock".to_string(), 1000);
        assert!(lock.acquire("owner1".to_string()).unwrap());
        assert!(!lock.acquire("owner2".to_string()).unwrap());
        assert!(lock.release("owner1".to_string()).unwrap());
    }
}












