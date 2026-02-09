//! Macros for common operations
//!
//! Provides convenient macros for common patterns.

/// Macro for creating a PyResult with error handling
#[macro_export]
macro_rules! py_result {
    ($expr:expr) => {
        $expr.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
    };
}

/// Macro for unwrapping with Python error
#[macro_export]
macro_rules! py_unwrap {
    ($expr:expr) => {
        $expr.map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?
    };
}

/// Macro for timing operations
#[macro_export]
macro_rules! time_it {
    ($name:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed();
        log::debug!("{} took {:?}", $name, elapsed);
        result
    }};
}

/// Macro for creating Python dict from key-value pairs
#[macro_export]
macro_rules! py_dict {
    ($py:expr, $( $key:expr => $val:expr ),* $(,)?) => {{
        use pyo3::types::PyDict;
        let dict = PyDict::new($py);
        $(
            dict.set_item($key, $val)?;
        )*
        dict
    }};
}

/// Macro for creating error with context
#[macro_export]
macro_rules! py_error {
    ($msg:expr) => {
        pyo3::exceptions::PyRuntimeError::new_err($msg)
    };
    ($msg:expr, $($arg:expr),*) => {
        pyo3::exceptions::PyRuntimeError::new_err(format!($msg, $($arg),*))
    };
}

/// Macro for logging with Python integration
#[macro_export]
macro_rules! py_log {
    (debug, $($arg:tt)*) => {
        log::debug!($($arg)*);
    };
    (info, $($arg:tt)*) => {
        log::info!($($arg)*);
    };
    (warn, $($arg:tt)*) => {
        log::warn!($($arg)*);
    };
    (error, $($arg:tt)*) => {
        log::error!($($arg)*);
    };
}

/// Macro for creating PyResult from Option
#[macro_export]
macro_rules! option_to_py_result {
    ($opt:expr, $err_msg:expr) => {
        $opt.ok_or_else(|| pyo3::exceptions::PyValueError::new_err($err_msg))
    };
}

/// Macro for validating input
#[macro_export]
macro_rules! validate {
    ($condition:expr, $msg:expr) => {
        if !$condition {
            return Err(pyo3::exceptions::PyValueError::new_err($msg));
        }
    };
}

/// Macro for creating benchmark
#[macro_export]
macro_rules! benchmark {
    ($name:expr, $iterations:expr, $block:block) => {{
        let mut times = Vec::new();
        for _ in 0..$iterations {
            let start = std::time::Instant::now();
            $block;
            times.push(start.elapsed());
        }
        let avg: f64 = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;
        let min = times.iter().min().unwrap();
        let max = times.iter().max().unwrap();
        log::info!("Benchmark {}: avg={:?}, min={:?}, max={:?}", $name, avg, min, max);
        (avg, min, max)
    }};
}












