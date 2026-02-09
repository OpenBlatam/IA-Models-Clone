//! KV Cache Service - High-performance distributed KV cache
//!
//! This service provides a distributed key-value cache optimized for
//! transformer model KV cache storage.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::RwLock as AsyncRwLock;

/// KV Cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    data: Vec<u8>,
    timestamp: u64,
}

/// Cache storage
type CacheStorage = Arc<AsyncRwLock<HashMap<String, CacheEntry>>>;

/// Service state
#[derive(Clone)]
struct AppState {
    cache: CacheStorage,
}

/// Get cache entry
async fn get_cache(
    Path((layer, position)): Path<(usize, usize)>,
    State(state): State<AppState>,
) -> Result<Json<Vec<u8>>, StatusCode> {
    let key = format!("{}:{}", layer, position);
    let cache = state.cache.read().await;
    
    match cache.get(&key) {
        Some(entry) => Ok(Json(entry.data.clone())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Put cache entry
async fn put_cache(
    Path((layer, position)): Path<(usize, usize)>,
    State(state): State<AppState>,
    Json(data): Json<Vec<u8>>,
) -> StatusCode {
    let key = format!("{}:{}", layer, position);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let entry = CacheEntry {
        data,
        timestamp,
    };
    
    let mut cache = state.cache.write().await;
    cache.insert(key, entry);
    
    StatusCode::OK
}

/// Delete cache entry
async fn delete_cache(
    Path((layer, position)): Path<(usize, usize)>,
    State(state): State<AppState>,
) -> StatusCode {
    let key = format!("{}:{}", layer, position);
    let mut cache = state.cache.write().await;
    
    if cache.remove(&key).is_some() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

/// Health check
async fn health() -> Json<HashMap<&'static str, &'static str>> {
    let mut response = HashMap::new();
    response.insert("status", "healthy");
    response.insert("service", "kv-cache-service");
    Json(response)
}

#[tokio::main]
async fn main() {
    // Initialize state
    let state = AppState {
        cache: Arc::new(AsyncRwLock::new(HashMap::new())),
    };
    
    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/cache/:layer/:position", get(get_cache))
        .route("/v1/cache/:layer/:position", put(put_cache))
        .route("/v1/cache/:layer/:position", delete(delete_cache))
        .with_state(state);
    
    // Run server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8010")
        .await
        .unwrap();
    
    println!("KV Cache Service listening on http://0.0.0.0:8010");
    axum::serve(listener, app).await.unwrap();
}












