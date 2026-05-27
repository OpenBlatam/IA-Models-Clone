pub mod mm;
pub mod vfs;
pub mod sched;
pub mod syscalls;

pub use syscalls::{SysCall, Response, TelemetryInfo};
pub use mm::MemoryManager;
pub use vfs::VirtualFileSystem;
pub use sched::Scheduler;

use parking_lot::RwLock;
use std::sync::Arc;

pub struct KernelCore {
    pub mm: Arc<MemoryManager>,
    pub vfs: RwLock<VirtualFileSystem>,
    pub sched: Arc<Scheduler>,
}

impl KernelCore {
    pub fn new() -> Self {
        KernelCore {
            mm: Arc::new(MemoryManager::new("nexus_memory.json")),
            vfs: RwLock::new(VirtualFileSystem::new("nexus_vfs_jail")),
            sched: Arc::new(Scheduler::new()),
        }
    }
    
    pub fn should_break_circuit(&self, call_type: &str, payload: &str) -> bool {
        if call_type != "SYS_EXEC" { return false; }
        let hash = self.mm.hash_payload(call_type, payload);
        if let Some(mem) = self.mm.meta_learner_memory.get(&hash) {
            if mem.failure_count >= 3 && mem.success_count == 0 {
                return true;
            }
        }
        false
    }
}
