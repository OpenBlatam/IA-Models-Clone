use std::path::PathBuf;

pub struct VirtualFileSystem {
    pub jail_root: PathBuf,
}

impl VirtualFileSystem {
    pub fn new(jail_dir: &str) -> Self {
        let path = PathBuf::from(jail_dir);
        if !path.exists() {
            std::fs::create_dir_all(&path).expect("Failed to create VFS jail");
        }
        VirtualFileSystem { jail_root: path.canonicalize().unwrap() }
    }

    fn is_safe_path(&self, requested: &str) -> Option<PathBuf> {
        let req_path = self.jail_root.join(requested);
        if let Ok(canon) = req_path.canonicalize() {
            if canon.starts_with(&self.jail_root) {
                return Some(canon);
            }
        } else {
            if let Some(parent) = req_path.parent() {
                if let Ok(canon_parent) = parent.canonicalize() {
                    if canon_parent.starts_with(&self.jail_root) {
                        return Some(req_path);
                    }
                }
            }
        }
        None
    }

    pub fn read_file(&self, path: &str) -> Result<String, String> {
        if let Some(safe_path) = self.is_safe_path(path) {
            std::fs::read_to_string(safe_path).map_err(|e| e.to_string())
        } else {
            Err("EPERM: Permission Denied. Attempted to escape VFS Jail.".to_string())
        }
    }

    pub fn write_file(&self, path: &str, content: &str) -> Result<(), String> {
        if let Some(safe_path) = self.is_safe_path(path) {
            std::fs::write(safe_path, content).map_err(|e| e.to_string())
        } else {
            Err("EPERM: Permission Denied. Attempted to escape VFS Jail.".to_string())
        }
    }
}
