//! Feature Flags
//!
//! Provides feature flag management.

use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Feature flag
#[derive(Debug, Clone)]
struct FeatureFlag {
    name: String,
    enabled: bool,
    rollout_percentage: f64,
    conditions: HashMap<String, String>,
}

/// Feature flag manager
#[pyclass]
pub struct FeatureFlagManager {
    flags: Arc<Mutex<HashMap<String, FeatureFlag>>>,
    stats: Arc<Mutex<FeatureFlagStats>>,
}

#[derive(Debug, Default)]
struct FeatureFlagStats {
    total_checks: usize,
    enabled_checks: usize,
    disabled_checks: usize,
}

#[pymethods]
impl FeatureFlagManager {
    #[new]
    pub fn new() -> Self {
        Self {
            flags: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(FeatureFlagStats::default())),
        }
    }

    pub fn register_flag(
        &self,
        name: String,
        enabled: bool,
        rollout_percentage: Option<f64>,
        conditions: Option<PyObject>,
    ) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut cond_map = HashMap::new();
            if let Some(cond_obj) = conditions {
                if let Ok(dict) = cond_obj.downcast::<PyDict>(py) {
                    for (key, value) in dict.iter() {
                        if let (Ok(k), Ok(v)) = (key.extract::<String>(), value.extract::<String>()) {
                            cond_map.insert(k, v);
                        }
                    }
                }
            }
            
            let flag = FeatureFlag {
                name: name.clone(),
                enabled,
                rollout_percentage: rollout_percentage.unwrap_or(100.0),
                conditions: cond_map,
            };
            
            self.flags.lock().unwrap().insert(name, flag);
            Ok(())
        })
    }

    pub fn is_enabled(&self, name: String, context: Option<PyObject>) -> PyResult<bool> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_checks += 1;
        
        let flags = self.flags.lock().unwrap();
        if let Some(flag) = flags.get(&name) {
            if !flag.enabled {
                stats.disabled_checks += 1;
                return Ok(false);
            }
            
            // Check rollout percentage
            let hash = self.hash_context(&name, context.as_ref());
            let percentage = (hash % 100) as f64;
            if percentage >= flag.rollout_percentage {
                stats.disabled_checks += 1;
                return Ok(false);
            }
            
            // Check conditions
            if !flag.conditions.is_empty() {
                if let Some(ctx) = context {
                    Python::with_gil(|py| {
                        if let Ok(ctx_dict) = ctx.downcast::<PyDict>(py) {
                            for (key, value) in &flag.conditions {
                                if let Ok(ctx_value) = ctx_dict.get_item(key) {
                                    if let Ok(ctx_str) = ctx_value.and_then(|v| v.extract::<String>()) {
                                        if ctx_str != *value {
                                            stats.disabled_checks += 1;
                                            return Ok(false);
                                        }
                                    }
                                }
                            }
                        }
                        Ok(())
                    })?;
                } else {
                    stats.disabled_checks += 1;
                    return Ok(false);
                }
            }
            
            stats.enabled_checks += 1;
            Ok(true)
        } else {
            stats.disabled_checks += 1;
            Ok(false)
        }
    }

    pub fn enable_flag(&self, name: String) -> PyResult<()> {
        let mut flags = self.flags.lock().unwrap();
        if let Some(flag) = flags.get_mut(&name) {
            flag.enabled = true;
        }
        Ok(())
    }

    pub fn disable_flag(&self, name: String) -> PyResult<()> {
        let mut flags = self.flags.lock().unwrap();
        if let Some(flag) = flags.get_mut(&name) {
            flag.enabled = false;
        }
        Ok(())
    }

    pub fn set_rollout(&self, name: String, percentage: f64) -> PyResult<()> {
        let mut flags = self.flags.lock().unwrap();
        if let Some(flag) = flags.get_mut(&name) {
            flag.rollout_percentage = percentage.min(100.0).max(0.0);
        }
        Ok(())
    }

    pub fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.stats.lock().unwrap();
            let flags = self.flags.lock().unwrap();
            let dict = PyDict::new(py);
            dict.set_item("total_checks", stats.total_checks)?;
            dict.set_item("enabled_checks", stats.enabled_checks)?;
            dict.set_item("disabled_checks", stats.disabled_checks)?;
            dict.set_item("total_flags", flags.len())?;
            dict.set_item("enabled_flags", flags.values().filter(|f| f.enabled).count())?;
            Ok(dict.into())
        })
    }

    fn hash_context(&self, name: &str, context: Option<&PyObject>) -> u64 {
        // Simple hash for rollout percentage
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        if let Some(ctx) = context {
            Python::with_gil(|py| {
                if let Ok(dict) = ctx.downcast::<PyDict>(py) {
                    for (key, value) in dict.iter() {
                        if let (Ok(k), Ok(v)) = (key.extract::<String>(), value.extract::<String>()) {
                            k.hash(&mut hasher);
                            v.hash(&mut hasher);
                        }
                    }
                }
            });
        }
        hasher.finish()
    }
}

#[pyfunction]
pub fn create_feature_flag_manager() -> FeatureFlagManager {
    FeatureFlagManager::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_flag_manager() {
        let manager = FeatureFlagManager::new();
        manager.register_flag("test_flag".to_string(), true, Some(100.0), None).unwrap();
        assert!(manager.is_enabled("test_flag".to_string(), None).unwrap());
    }
}












