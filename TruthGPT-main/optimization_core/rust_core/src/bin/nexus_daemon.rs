use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use tracing::{info, warn, error, debug};
use tracing_subscriber;

// Use the new nexus_kernel module layout
use truthgpt_rust::nexus_kernel::*;

async fn handle_syscall(syscall: SysCall, kernel: Arc<KernelCore>) -> Response {
    let start_time = Instant::now();
    let pid_clone = syscall.pid.clone();

    // 1. Register Process
    kernel.sched.register_process(syscall.pid.clone());

    // 2. Circuit Breaker
    if kernel.should_break_circuit(&syscall.call_type, &syscall.payload) {
        warn!("🛡️ CIRCUIT BREAKER: Bloqueando {} destructivo (PID: {})", syscall.call_type, syscall.pid);
        return Response {
            status: "circuit_broken".to_string(),
            message: "⚠️ KERNEL CIRCUIT BREAKER: Bloqueado por seguridad.".to_string(),
            telemetry: None,
        };
    }

    let mut success = true;
    let mut msg = String::new();

    // 3. SysCall Routing
    match syscall.call_type.as_str() {
        "SYS_PING" => msg = "Pong! Refactored Linux-like AI-OS Kernel alive.".to_string(),
        
        "SYS_MEM_WRITE" => {
            #[cfg(feature = "simd-json")]
            let parsed = simd_json::from_str::<serde_json::Value>(&mut syscall.payload.clone().into_bytes());
            #[cfg(not(feature = "simd-json"))]
            let parsed = serde_json::from_str::<serde_json::Value>(&syscall.payload);

            if let Ok(p) = parsed {
                if let (Some(k), Some(v)) = (p["key"].as_str(), p["value"].as_str()) {
                    kernel.mm.virtual_context_memory.insert(k.to_string(), v.to_string());
                    msg = "L1 Cache write successful".to_string();
                } else {
                    success = false; msg = "Invalid format".to_string();
                }
            } else {
                success = false; msg = "Invalid JSON".to_string();
            }
        },
        "SYS_MEM_READ" => {
            if let Some(val) = kernel.mm.virtual_context_memory.get(syscall.payload.trim()) {
                msg = val.value().clone();
            } else {
                success = false; msg = "Key not found".to_string();
            }
        },

        "SYS_FILE_READ" => {
            match kernel.vfs.read().read_file(&syscall.payload) {
                Ok(content) => msg = content,
                Err(e) => { success = false; msg = e; }
            }
        },
        "SYS_FILE_WRITE" => {
            #[cfg(feature = "simd-json")]
            let parsed = simd_json::from_str::<serde_json::Value>(&mut syscall.payload.clone().into_bytes());
            #[cfg(not(feature = "simd-json"))]
            let parsed = serde_json::from_str::<serde_json::Value>(&syscall.payload);

            if let Ok(p) = parsed {
                if let (Some(path), Some(c)) = (p["path"].as_str(), p["content"].as_str()) {
                    match kernel.vfs.read().write_file(path, c) {
                        Ok(_) => msg = "File written safely in VFS Jail".to_string(),
                        Err(e) => { success = false; msg = e; }
                    }
                } else {
                    success = false; msg = "Invalid format".to_string();
                }
            } else {
                success = false; msg = "Invalid JSON".to_string();
            }
        },
        
        "SYS_KILL" => {
            match kernel.sched.kill_process(&syscall.payload) {
                Ok(_) => msg = format!("SIGKILL sent to PID {}", syscall.payload),
                Err(e) => { success = false; msg = e; }
            }
        },

        "SYS_EXEC" => {
            info!("🚀 [Zero-Trust] Ejecutando código aislado (PID: {})", syscall.pid);
            
            if !kernel.sched.is_alive(&syscall.pid) {
                return Response {
                    status: "error".to_string(),
                    message: "Process received SIGKILL before execution".to_string(),
                    telemetry: None,
                };
            }

            let cmd_future = Command::new("python").arg("-c").arg(&syscall.payload).output();

            match timeout(Duration::from_secs(10), cmd_future).await {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    if output.status.success() {
                        msg = if stdout.is_empty() { "Ejecución finalizada.".to_string() } else { stdout };
                    } else {
                        success = false; msg = format!("Ejecución fallida:\n{}", stderr);
                    }
                }
                Ok(Err(e)) => { success = false; msg = format!("Fallo al levantar subproceso: {}", e); }
                Err(_) => {
                    success = false;
                    msg = "🚨 KERNEL PANIC: Subproceso excedió 10s (SIGKILL automático).".to_string();
                    error!("💀 [Zero-Trust] Asesinó el subproceso PID {} (Timeout).", syscall.pid);
                }
            }
        },
        _ => { success = false; msg = format!("Unknown SysCall: {}", syscall.call_type); }
    }

    let duration = start_time.elapsed().as_millis();
    let mem_state = kernel.mm.record_result(&syscall.call_type, &syscall.payload, success);
    
    // Process End
    kernel.sched.active_processes.remove(&pid_clone);

    Response {
        status: if success { "success".to_string() } else { "error".to_string() },
        message: msg,
        telemetry: Some(TelemetryInfo {
            execution_ms: duration,
            failures: mem_state.failure_count,
            circuit_broken: false,
        }),
    }
}

async fn handle_client(mut stream: TcpStream, kernel: Arc<KernelCore>) {
    let mut buffer = [0; 8192];
    match stream.read(&mut buffer).await {
        Ok(size) if size > 0 => {
            #[cfg(feature = "simd-json")]
            let req: Result<SysCall, _> = simd_json::from_slice(&mut buffer[..size]);
            #[cfg(not(feature = "simd-json"))]
            let req: Result<SysCall, _> = serde_json::from_slice(&buffer[..size]);
            
            let response = match req {
                Ok(syscall) => {
                    debug!("🧠 [PID: {}] Priority {} Requested {}", syscall.pid, syscall.priority, syscall.call_type);
                    handle_syscall(syscall, kernel).await
                }
                Err(e) => {
                    warn!("Failed to parse SysCall: {}", e);
                    Response {
                        status: "error".to_string(),
                        message: "Invalid Syscall struct.".to_string(),
                        telemetry: None,
                    }
                }
            };
            
            #[cfg(feature = "simd-json")]
            let response_bytes = simd_json::to_vec(&response).unwrap();
            #[cfg(not(feature = "simd-json"))]
            let response_bytes = serde_json::to_vec(&response).unwrap();
            
            let _ = stream.write_all(&response_bytes).await;
        }
        _ => {}
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();

    let host = "127.0.0.1:50051";
    let listener = TcpListener::bind(host).await.expect("Failed to bind");
    let kernel = Arc::new(KernelCore::new());
    
    info!("🚀 Refactored Linux-like AI-OS Kernel started. Listening on {}...", host);
    info!("🧬 Structure: Modular Multi-File /src/nexus_kernel/ (mm, vfs, sched, syscalls)");

    loop {
        if let Ok((stream, _)) = listener.accept().await {
            let kernel_clone = Arc::clone(&kernel);
            tokio::spawn(async move {
                handle_client(stream, kernel_clone).await;
            });
        }
    }
}
