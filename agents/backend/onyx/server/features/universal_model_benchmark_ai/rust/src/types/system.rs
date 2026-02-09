//! System Information Types
//!
//! Types for system and version information.

use serde::{Deserialize, Serialize};

/// Version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub name: String,
    pub description: String,
}

impl VersionInfo {
    /// Create new version info.
    pub fn new(version: String, name: String, description: String) -> Self {
        Self {
            version,
            name,
            description,
        }
    }
    
    /// Create from environment.
    pub fn from_env() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: env!("CARGO_PKG_NAME").to_string(),
            description: env!("CARGO_PKG_DESCRIPTION").to_string(),
        }
    }
    
    /// Get formatted version string.
    pub fn to_string(&self) -> String {
        format!("{} v{}", self.name, self.version)
    }
}

/// System information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub version: String,
    pub name: String,
    pub description: String,
    pub hostname: Option<String>,
    pub features: Vec<String>,
}

impl SystemInfo {
    /// Create new system info.
    pub fn new(
        version: String,
        name: String,
        description: String,
    ) -> Self {
        Self {
            version,
            name,
            description,
            hostname: std::env::var("HOSTNAME").ok(),
            features: Vec::new(),
        }
    }
    
    /// Create from environment.
    pub fn from_env() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            name: env!("CARGO_PKG_NAME").to_string(),
            description: env!("CARGO_PKG_DESCRIPTION").to_string(),
            hostname: std::env::var("HOSTNAME").ok(),
            features: Vec::new(),
        }
    }
    
    /// Add a feature.
    pub fn with_feature(mut self, feature: String) -> Self {
        self.features.push(feature);
        self
    }
    
    /// Add multiple features.
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features.extend(features);
        self
    }
    
    /// Check if a feature is enabled.
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
    
    /// Get formatted info string.
    pub fn to_string(&self) -> String {
        format!(
            "{} v{} on {}",
            self.name,
            self.version,
            self.hostname.as_deref().unwrap_or("unknown")
        )
    }
}




