//! Browser cache module using LocalStorage/IndexedDB

use wasm_bindgen::prelude::*;
use web_sys::{Storage, window};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CacheEntry {
    pub key: String,
    pub value: String,
    pub created_at: u64,
    pub expires_at: Option<u64>,
}

pub struct BrowserCache {
    prefix: String,
}

impl BrowserCache {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let storage = self.get_storage()?;
        let full_key = self.make_key(key);
        
        match storage.get_item(&full_key) {
            Ok(Some(json)) => {
                if let Ok(entry) = serde_json::from_str::<CacheEntry>(&json) {
                    if let Some(expires_at) = entry.expires_at {
                        if Self::current_time() > expires_at {
                            let _ = storage.remove_item(&full_key);
                            return None;
                        }
                    }
                    Some(entry.value)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn set(&self, key: &str, value: &str, ttl_seconds: Option<u64>) {
        if let Some(storage) = self.get_storage() {
            let full_key = self.make_key(key);
            let now = Self::current_time();
            
            let entry = CacheEntry {
                key: key.to_string(),
                value: value.to_string(),
                created_at: now,
                expires_at: ttl_seconds.map(|ttl| now + ttl * 1000),
            };
            
            if let Ok(json) = serde_json::to_string(&entry) {
                let _ = storage.set_item(&full_key, &json);
            }
        }
    }

    pub fn remove(&self, key: &str) -> bool {
        if let Some(storage) = self.get_storage() {
            let full_key = self.make_key(key);
            storage.remove_item(&full_key).is_ok()
        } else {
            false
        }
    }

    pub fn clear(&self) {
        if let Some(storage) = self.get_storage() {
            let len = storage.length().unwrap_or(0);
            let mut keys_to_remove = Vec::new();
            
            for i in 0..len {
                if let Ok(Some(key)) = storage.key(i) {
                    if key.starts_with(&self.prefix) {
                        keys_to_remove.push(key);
                    }
                }
            }
            
            for key in keys_to_remove {
                let _ = storage.remove_item(&key);
            }
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        
        if let Some(storage) = self.get_storage() {
            let len = storage.length().unwrap_or(0);
            
            for i in 0..len {
                if let Ok(Some(key)) = storage.key(i) {
                    if key.starts_with(&self.prefix) {
                        keys.push(key[self.prefix.len() + 1..].to_string());
                    }
                }
            }
        }
        
        keys
    }

    pub fn cleanup_expired(&self) -> usize {
        let mut removed = 0;
        
        if let Some(storage) = self.get_storage() {
            let now = Self::current_time();
            let len = storage.length().unwrap_or(0);
            let mut keys_to_remove = Vec::new();
            
            for i in 0..len {
                if let Ok(Some(key)) = storage.key(i) {
                    if key.starts_with(&self.prefix) {
                        if let Ok(Some(json)) = storage.get_item(&key) {
                            if let Ok(entry) = serde_json::from_str::<CacheEntry>(&json) {
                                if let Some(expires_at) = entry.expires_at {
                                    if now > expires_at {
                                        keys_to_remove.push(key);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            for key in keys_to_remove {
                if storage.remove_item(&key).is_ok() {
                    removed += 1;
                }
            }
        }
        
        removed
    }

    fn get_storage(&self) -> Option<Storage> {
        window()?.local_storage().ok().flatten()
    }

    fn make_key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }

    fn current_time() -> u64 {
        js_sys::Date::now() as u64
    }
}

impl Default for BrowserCache {
    fn default() -> Self {
        Self::new("transcriber")
    }
}

#[wasm_bindgen]
pub struct JsCache {
    inner: BrowserCache,
}

#[wasm_bindgen]
impl JsCache {
    #[wasm_bindgen(constructor)]
    pub fn new(prefix: &str) -> Self {
        Self {
            inner: BrowserCache::new(prefix),
        }
    }

    #[wasm_bindgen]
    pub fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key)
    }

    #[wasm_bindgen]
    pub fn set(&self, key: &str, value: &str, ttl_seconds: Option<u64>) {
        self.inner.set(key, value, ttl_seconds);
    }

    #[wasm_bindgen]
    pub fn remove(&self, key: &str) -> bool {
        self.inner.remove(key)
    }

    #[wasm_bindgen]
    pub fn clear(&self) {
        self.inner.clear();
    }

    #[wasm_bindgen]
    pub fn has(&self, key: &str) -> bool {
        self.inner.has(key)
    }

    #[wasm_bindgen]
    pub fn keys(&self) -> Vec<JsValue> {
        self.inner.keys().into_iter().map(JsValue::from).collect()
    }

    #[wasm_bindgen]
    pub fn cleanup_expired(&self) -> usize {
        self.inner.cleanup_expired()
    }
}












