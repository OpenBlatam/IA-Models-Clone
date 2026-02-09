//! State Machine
//!
//! Provides finite state machine implementation.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};

/// State transition
#[derive(Debug, Clone)]
struct Transition {
    from_state: String,
    to_state: String,
    event: String,
}

/// State machine
#[pyclass]
pub struct StateMachine {
    current_state: Arc<Mutex<String>>,
    initial_state: String,
    final_states: Arc<Mutex<HashSet<String>>>,
    transitions: Arc<Mutex<Vec<Transition>>>,
    history: Arc<Mutex<Vec<(String, String, String)>>>, // (from, to, event)
}

#[pymethods]
impl StateMachine {
    #[new]
    pub fn new(initial_state: String) -> Self {
        Self {
            current_state: Arc::new(Mutex::new(initial_state.clone())),
            initial_state,
            final_states: Arc::new(Mutex::new(HashSet::new())),
            transitions: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_transition(&self, from_state: String, to_state: String, event: String) -> PyResult<()> {
        let mut transitions = self.transitions.lock().unwrap();
        transitions.push(Transition {
            from_state,
            to_state,
            event,
        });
        Ok(())
    }

    pub fn add_final_state(&self, state: String) -> PyResult<()> {
        self.final_states.lock().unwrap().insert(state);
        Ok(())
    }

    pub fn transition(&self, event: String) -> PyResult<bool> {
        let mut current_state = self.current_state.lock().unwrap();
        let transitions = self.transitions.lock().unwrap();
        let mut history = self.history.lock().unwrap();
        
        // Find valid transition
        if let Some(transition) = transitions.iter().find(|t| {
            t.from_state == *current_state && t.event == event
        }) {
            let from = current_state.clone();
            *current_state = transition.to_state.clone();
            history.push((from, transition.to_state.clone(), event));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_current_state(&self) -> String {
        self.current_state.lock().unwrap().clone()
    }

    pub fn is_final_state(&self) -> bool {
        let current_state = self.current_state.lock().unwrap();
        self.final_states.lock().unwrap().contains(&*current_state)
    }

    pub fn reset(&self) -> PyResult<()> {
        let mut current_state = self.current_state.lock().unwrap();
        *current_state = self.initial_state.clone();
        self.history.lock().unwrap().clear();
        Ok(())
    }

    pub fn get_history(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let history = self.history.lock().unwrap();
            let result: Vec<PyObject> = history.iter()
                .map(|(from, to, event)| {
                    let dict = PyDict::new(py);
                    dict.set_item("from", from).unwrap();
                    dict.set_item("to", to).unwrap();
                    dict.set_item("event", event).unwrap();
                    dict.into()
                })
                .collect();
            Ok(result)
        })
    }

    pub fn get_available_events(&self) -> PyResult<Vec<String>> {
        let current_state = self.current_state.lock().unwrap();
        let transitions = self.transitions.lock().unwrap();
        let events: Vec<String> = transitions.iter()
            .filter(|t| t.from_state == *current_state)
            .map(|t| t.event.clone())
            .collect();
        Ok(events)
    }
}

#[pyfunction]
pub fn create_state_machine(initial_state: String) -> StateMachine {
    StateMachine::new(initial_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine() {
        let sm = StateMachine::new("idle".to_string());
        sm.add_transition("idle".to_string(), "running".to_string(), "start".to_string()).unwrap();
        assert!(sm.transition("start".to_string()).unwrap());
        assert_eq!(sm.get_current_state(), "running");
    }
}












