use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActionMemory {
    pub payload_hash: String,
    pub success_count: u32,
    pub failure_count: u32,
}

pub struct MemoryManager {
    pub meta_learner_memory: DashMap<String, ActionMemory>,
    pub meta_memory_file: String,
    pub virtual_context_memory: DashMap<String, String>,
}

impl MemoryManager {
    pub fn new(file_path: &str) -> Self {
        let memory = DashMap::new();
        if let Ok(mut file) = File::open(file_path) {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(loaded) = serde_json::from_str::<std::collections::HashMap<String, ActionMemory>>(&contents) {
                    for (k, v) in loaded {
                        memory.insert(k, v);
                    }
                }
            }
        }
        MemoryManager {
            meta_learner_memory: memory,
            meta_memory_file: file_path.to_string(),
            virtual_context_memory: DashMap::new(),
        }
    }

    pub fn hash_payload(&self, call_type: &str, payload: &str) -> String {
        let content = format!("{}:{}", call_type, payload);
        let hash = content.chars().fold(0u64, |acc, c| acc.wrapping_add(c as u64));
        format!("{:x}", hash)
    }

    pub fn record_result(&self, call_type: &str, payload: &str, success: bool) -> ActionMemory {
        if call_type != "SYS_EXEC" {
            return ActionMemory { payload_hash: "".to_string(), success_count: 0, failure_count: 0 };
        }
        let hash = self.hash_payload(call_type, payload);
        
        let mut entry = self.meta_learner_memory.entry(hash.clone()).or_insert(ActionMemory {
            payload_hash: hash,
            success_count: 0,
            failure_count: 0,
        });
        
        if success {
            entry.success_count += 1;
            entry.failure_count = 0;
        } else {
            entry.failure_count += 1;
        }
        let cloned = entry.clone();
        drop(entry);
        
        self.save_meta();
        cloned
    }

    fn save_meta(&self) {
        if let Ok(mut file) = OpenOptions::new().write(true).create(true).truncate(true).open(&self.meta_memory_file) {
            let std_map: std::collections::HashMap<_, _> = self.meta_learner_memory.iter().map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone())).collect();
            if let Ok(json) = serde_json::to_string(&std_map) {
                let _ = file.write_all(json.as_bytes());
            }
        }
    }
}
