//! Context Management
//!
//! Provides context propagation and management.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Request context
#[pyclass]
pub struct RequestContext {
    request_id: String,
    user_id: Option<String>,
    metadata: Arc<Mutex<HashMap<String, String>>>,
    created_at: u64,
    expires_at: Option<u64>,
}

#[pymethods]
impl RequestContext {
    #[new]
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            user_id: None,
            metadata: Arc::new(Mutex::new(HashMap::new())),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expires_at: None,
        }
    }

    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.expires_at = Some(self.created_at + ttl_seconds);
        self
    }

    pub fn get_request_id(&self) -> String {
        self.request_id.clone()
    }

    pub fn get_user_id(&self) -> Option<String> {
        self.user_id.clone()
    }

    pub fn set_metadata(&self, key: String, value: String) -> PyResult<()> {
        self.metadata.lock().unwrap().insert(key, value);
        Ok(())
    }

    pub fn get_metadata(&self, key: String) -> Option<String> {
        self.metadata.lock().unwrap().get(&key).cloned()
    }

    pub fn get_all_metadata(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let metadata = self.metadata.lock().unwrap();
            let dict = PyDict::new(py);
            for (key, value) in metadata.iter() {
                dict.set_item(key, value)?;
            }
            Ok(dict.into())
        })
    }

    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            now >= expires_at
        } else {
            false
        }
    }

    pub fn get_age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now.saturating_sub(self.created_at)
    }
}

/// Context manager
#[pyclass]
pub struct ContextManager {
    contexts: Arc<Mutex<HashMap<String, RequestContext>>>,
    max_contexts: usize,
}

#[pymethods]
impl ContextManager {
    #[new]
    pub fn new(max_contexts: usize) -> Self {
        Self {
            contexts: Arc::new(Mutex::new(HashMap::new())),
            max_contexts,
        }
    }

    pub fn create_context(&self, request_id: String) -> PyResult<RequestContext> {
        let mut contexts = self.contexts.lock().unwrap();
        
        // Clean expired contexts
        contexts.retain(|_, ctx| !ctx.is_expired());
        
        // Check limit
        if contexts.len() >= self.max_contexts {
            return Err(PyValueError::new_err("Context limit reached"));
        }
        
        let context = RequestContext::new(request_id.clone());
        contexts.insert(request_id, context.clone());
        Ok(context)
    }

    pub fn get_context(&self, request_id: String) -> Option<RequestContext> {
        let contexts = self.contexts.lock().unwrap();
        contexts.get(&request_id).cloned()
    }

    pub fn remove_context(&self, request_id: String) -> PyResult<()> {
        let mut contexts = self.contexts.lock().unwrap();
        contexts.remove(&request_id);
        Ok(())
    }

    pub fn cleanup_expired(&self) -> usize {
        let mut contexts = self.contexts.lock().unwrap();
        let before = contexts.len();
        contexts.retain(|_, ctx| !ctx.is_expired());
        before - contexts.len()
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let contexts = self.contexts.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total", contexts.len())?;
            dict.set_item("max", self.max_contexts)?;
            dict.set_item("expired", contexts.values().filter(|ctx| ctx.is_expired()).count())?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
pub fn create_context_manager(max_contexts: Option<usize>) -> ContextManager {
    ContextManager::new(max_contexts.unwrap_or(1000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_context() {
        let ctx = RequestContext::new("req-1".to_string());
        assert_eq!(ctx.get_request_id(), "req-1");
        assert!(!ctx.is_expired());
    }

    #[test]
    fn test_context_manager() {
        let manager = ContextManager::new(100);
        let ctx = manager.create_context("req-1".to_string()).unwrap();
        assert_eq!(ctx.get_request_id(), "req-1");
    }
}












