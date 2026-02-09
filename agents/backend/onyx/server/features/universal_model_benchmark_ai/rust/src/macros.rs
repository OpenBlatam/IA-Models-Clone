//! Useful macros for common operations.
//!
//! Provides macros to reduce boilerplate and improve code ergonomics.

/// Macro to create a BenchmarkConfig quickly.
///
/// # Example
///
/// ```rust,no_run
/// use benchmark_core::benchmark_config;
///
/// let config = benchmark_config! {
///     model_path: "path/to/model",
///     batch_size: 32,
///     max_tokens: 512,
/// }?;
/// ```
#[macro_export]
macro_rules! benchmark_config {
    (
        model_path: $path:expr
        $(, $field:ident: $value:expr)*
    ) => {
        {
            let mut builder = $crate::BenchmarkConfig::builder()
                .model_path($path.to_string());
            $(
                builder = builder.$field($value);
            )*
            builder.build()
        }
    };
}

/// Macro to create Metrics quickly.
///
/// # Example
///
/// ```rust,no_run
/// use benchmark_core::metrics;
///
/// let m = metrics! {
///     accuracy: 0.95,
///     latency_p50: 0.1,
///     throughput: 100.0,
/// };
/// ```
#[macro_export]
macro_rules! metrics {
    (
        $($field:ident: $value:expr),* $(,)?
    ) => {
        $crate::Metrics {
            $($field: $value,)*
            ..$crate::types::Metrics::default()
        }
    };
}

/// Macro to create a Result with context.
///
/// # Example
///
/// ```rust,no_run
/// use benchmark_core::bail;
///
/// if value < 0 {
///     bail!("Value must be positive, got {}", value);
/// }
/// ```
#[macro_export]
macro_rules! bail {
    ($msg:literal $(, $arg:expr)*) => {
        return Err($crate::error::BenchmarkError::Other(format!($msg $(, $arg)*)))
    };
}

/// Macro to ensure a condition is true.
///
/// # Example
///
/// ```rust,no_run
/// use benchmark_core::ensure;
///
/// ensure!(value > 0, "Value must be positive");
/// ```
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err($crate::error::BenchmarkError::invalid_input($msg));
        }
    };
    ($cond:expr, $fmt:literal, $($arg:expr),*) => {
        if !$cond {
            return Err($crate::error::BenchmarkError::invalid_input(format!($fmt, $($arg),*)));
        }
    };
}

/// Macro to format error messages consistently.
#[macro_export]
macro_rules! format_err {
    ($kind:ident, $msg:literal $(, $arg:expr)*) => {
        $crate::error::BenchmarkError::$kind(format!($msg $(, $arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_config_macro() {
        let config = benchmark_config! {
            model_path: "test",
            batch_size: 32,
        };
        assert!(config.is_ok());
    }
    
    #[test]
    fn test_metrics_macro() {
        let m = metrics! {
            accuracy: 0.95,
            latency_p50: 0.1,
        };
        assert_eq!(m.accuracy, 0.95);
        assert_eq!(m.latency_p50, 0.1);
    }
}

