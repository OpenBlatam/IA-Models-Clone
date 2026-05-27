use dashmap::DashMap;

pub struct Scheduler {
    pub active_processes: DashMap<String, bool>, // PID -> IsAlive
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            active_processes: DashMap::new(),
        }
    }

    pub fn register_process(&self, pid: String) {
        self.active_processes.insert(pid, true);
    }

    pub fn kill_process(&self, pid: &str) -> Result<(), String> {
        if self.active_processes.contains_key(pid) {
            self.active_processes.insert(pid.to_string(), false); // Marked for death
            Ok(())
        } else {
            Err(format!("ESRCH: No such process (PID: {})", pid))
        }
    }
    
    pub fn is_alive(&self, pid: &str) -> bool {
        if let Some(r) = self.active_processes.get(pid) {
            *r
        } else {
            self.active_processes.insert(pid.to_string(), true);
            true
        }
    }
}
