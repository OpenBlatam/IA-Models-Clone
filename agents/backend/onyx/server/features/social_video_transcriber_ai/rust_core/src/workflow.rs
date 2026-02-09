//! Workflow Engine
//!
//! Provides workflow orchestration and execution.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};

/// Workflow step
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    id: String,
    name: String,
    step_type: String,
    depends_on: Vec<String>,
    config: HashMap<String, String>,
}

/// Workflow state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl WorkflowState {
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkflowState::Pending => "pending",
            WorkflowState::Running => "running",
            WorkflowState::Completed => "completed",
            WorkflowState::Failed => "failed",
            WorkflowState::Cancelled => "cancelled",
        }
    }
}

/// Workflow execution
#[pyclass]
pub struct Workflow {
    id: String,
    steps: Arc<Mutex<HashMap<String, WorkflowStep>>>,
    execution_order: Arc<Mutex<VecDeque<String>>>,
    state: Arc<Mutex<WorkflowState>>,
    current_step: Arc<Mutex<Option<String>>>,
    results: Arc<Mutex<HashMap<String, PyObject>>>,
}

#[pymethods]
impl Workflow {
    #[new]
    pub fn new(workflow_id: String) -> Self {
        Self {
            id: workflow_id,
            steps: Arc::new(Mutex::new(HashMap::new())),
            execution_order: Arc::new(Mutex::new(VecDeque::new())),
            state: Arc::new(Mutex::new(WorkflowState::Pending)),
            current_step: Arc::new(Mutex::new(None)),
            results: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn add_step(
        &self,
        step_id: String,
        name: String,
        step_type: String,
        depends_on: Option<Vec<String>>,
        config: Option<PyObject>,
    ) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut config_map = HashMap::new();
            if let Some(config_obj) = config {
                if let Ok(dict) = config_obj.downcast::<PyDict>(py) {
                    for (key, value) in dict.iter() {
                        if let (Ok(k), Ok(v)) = (key.extract::<String>(), value.extract::<String>()) {
                            config_map.insert(k, v);
                        }
                    }
                }
            }
            
            let step = WorkflowStep {
                id: step_id.clone(),
                name,
                step_type,
                depends_on: depends_on.unwrap_or_default(),
                config: config_map,
            };
            
            self.steps.lock().unwrap().insert(step_id, step);
            Ok(())
        })
    }

    pub fn build_execution_order(&self) -> PyResult<Vec<String>> {
        let steps = self.steps.lock().unwrap();
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut in_progress = std::collections::HashSet::new();
        
        fn visit(
            step_id: &String,
            steps: &HashMap<String, WorkflowStep>,
            visited: &mut std::collections::HashSet<String>,
            in_progress: &mut std::collections::HashSet<String>,
            order: &mut Vec<String>,
        ) -> PyResult<()> {
            if visited.contains(step_id) {
                return Ok(());
            }
            
            if in_progress.contains(step_id) {
                return Err(PyValueError::new_err(format!("Circular dependency detected: {}", step_id)));
            }
            
            in_progress.insert(step_id.clone());
            
            if let Some(step) = steps.get(step_id) {
                for dep in &step.depends_on {
                    visit(dep, steps, visited, in_progress, order)?;
                }
            }
            
            in_progress.remove(step_id);
            visited.insert(step_id.clone());
            order.push(step_id.clone());
            Ok(())
        }
        
        for step_id in steps.keys() {
            visit(step_id, &steps, &mut visited, &mut in_progress, &mut order)?;
        }
        
        *self.execution_order.lock().unwrap() = order.iter().cloned().collect();
        Ok(order)
    }

    pub fn get_state(&self) -> String {
        self.state.lock().unwrap().as_str().to_string()
    }

    pub fn get_current_step(&self) -> Option<String> {
        self.current_step.lock().unwrap().clone()
    }

    pub fn set_step_result(&self, step_id: String, result: PyObject) -> PyResult<()> {
        self.results.lock().unwrap().insert(step_id, result);
        Ok(())
    }

    pub fn get_step_result(&self, step_id: String) -> Option<PyObject> {
        self.results.lock().unwrap().get(&step_id).cloned()
    }

    pub fn get_all_results(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let results = self.results.lock().unwrap();
            let dict = PyDict::new(py);
            for (key, value) in results.iter() {
                dict.set_item(key, value)?;
            }
            Ok(dict.into())
        })
    }
}

#[pyfunction]
pub fn create_workflow(workflow_id: String) -> Workflow {
    Workflow::new(workflow_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow() {
        let workflow = Workflow::new("test-workflow".to_string());
        assert_eq!(workflow.get_state(), "pending");
    }
}












