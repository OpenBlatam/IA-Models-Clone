use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SysCall {
    pub pid: String,
    pub priority: u8,
    pub call_type: String, // SYS_PING, SYS_EXEC, SYS_MEM_READ/WRITE, SYS_FILE_READ/WRITE, SYS_KILL
    pub payload: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Response {
    pub status: String,
    pub message: String,
    pub telemetry: Option<TelemetryInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TelemetryInfo {
    pub execution_ms: u128,
    pub failures: u32,
    pub circuit_broken: bool,
}
