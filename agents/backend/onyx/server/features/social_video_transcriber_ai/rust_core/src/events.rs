//! Event System
//!
//! Provides an event-driven architecture for decoupled communication.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::any::Any;

/// Event type
#[derive(Debug, Clone)]
pub enum EventType {
    CacheHit,
    CacheMiss,
    CacheEviction,
    BatchStart,
    BatchComplete,
    BatchError,
    CompressionStart,
    CompressionComplete,
    SearchStart,
    SearchComplete,
    Custom(String),
}

impl EventType {
    pub fn as_str(&self) -> &str {
        match self {
            EventType::CacheHit => "cache_hit",
            EventType::CacheMiss => "cache_miss",
            EventType::CacheEviction => "cache_eviction",
            EventType::BatchStart => "batch_start",
            EventType::BatchComplete => "batch_complete",
            EventType::BatchError => "batch_error",
            EventType::CompressionStart => "compression_start",
            EventType::CompressionComplete => "compression_complete",
            EventType::SearchStart => "search_start",
            EventType::SearchComplete => "search_complete",
            EventType::Custom(s) => s,
        }
    }
}

/// Event data
#[derive(Debug, Clone)]
pub struct Event {
    pub event_type: EventType,
    pub data: HashMap<String, PyObject>,
    pub timestamp: u64,
}

impl Event {
    pub fn new(event_type: EventType) -> Self {
        Self {
            event_type,
            data: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    pub fn with_data(mut self, key: String, value: PyObject) -> Self {
        self.data.insert(key, value);
        self
    }
}

/// Event handler trait
pub trait EventHandler: Send + Sync {
    fn handle(&self, event: &Event) -> PyResult<()>;
}

/// Event bus for managing events
#[pyclass]
pub struct EventBus {
    handlers: Arc<Mutex<HashMap<String, Vec<Box<dyn EventHandler>>>>>,
}

#[pymethods]
impl EventBus {
    #[new]
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn emit(&self, event: Event) -> PyResult<()> {
        let event_type = event.event_type.as_str().to_string();
        let handlers = self.handlers.lock().unwrap();
        
        if let Some(handlers_list) = handlers.get(&event_type) {
            for handler in handlers_list {
                handler.handle(&event)?;
            }
        }
        
        Ok(())
    }

    pub fn on(&self, event_type: String, handler: PyObject) -> PyResult<()> {
        // Note: In a real implementation, you'd need to wrap Python callables
        // For now, this is a placeholder structure
        Ok(())
    }

    pub fn off(&self, event_type: String) -> PyResult<()> {
        let mut handlers = self.handlers.lock().unwrap();
        handlers.remove(&event_type);
        Ok(())
    }

    pub fn clear(&self) -> PyResult<()> {
        let mut handlers = self.handlers.lock().unwrap();
        handlers.clear();
        Ok(())
    }

    pub fn get_subscribed_events(&self) -> PyResult<Vec<String>> {
        let handlers = self.handlers.lock().unwrap();
        Ok(handlers.keys().cloned().collect())
    }
}

#[pyfunction]
pub fn create_event_bus() -> EventBus {
    EventBus::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = Event::new(EventType::CacheHit);
        assert_eq!(event.event_type.as_str(), "cache_hit");
    }

    #[test]
    fn test_event_bus() {
        let bus = EventBus::new();
        let event = Event::new(EventType::CacheHit);
        assert!(bus.emit(event).is_ok());
    }
}












