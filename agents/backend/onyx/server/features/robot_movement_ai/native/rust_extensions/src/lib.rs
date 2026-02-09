/*
 * Rust Extensions for Robot Movement AI
 * =====================================
 * 
 * Extensiones Rust para operaciones de alto rendimiento:
 * - Parsing JSON ultra-rápido
 * - Operaciones de strings
 * - Algoritmos de búsqueda
 * - Operaciones de I/O
 */

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json;
use std::collections::HashMap;

/// Parsing JSON ultra-rápido usando serde_json
#[pyfunction]
fn fast_json_parse(json_str: &str) -> PyResult<PyObject> {
    let value: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("JSON parse error: {}", e)
        ))?;
    
    Python::with_gil(|py| {
        json_value_to_python(py, &value)
    })
}

/// Convertir valor JSON a objeto Python
fn json_value_to_python(py: Python, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(n.to_string().to_object(py))
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_python(py, item)?)?;
            }
            Ok(list.to_object(py))
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, json_value_to_python(py, v)?)?;
            }
            Ok(dict.to_object(py))
        }
    }
}

/// Búsqueda rápida en strings
#[pyfunction]
fn fast_string_search(text: &str, pattern: &str) -> PyResult<Vec<usize>> {
    let mut positions = Vec::new();
    let pattern_bytes = pattern.as_bytes();
    let text_bytes = text.as_bytes();
    
    if pattern_bytes.is_empty() {
        return Ok(positions);
    }
    
    // Algoritmo de búsqueda simple (se puede mejorar con KMP o Boyer-Moore)
    for i in 0..=text_bytes.len().saturating_sub(pattern_bytes.len()) {
        if text_bytes[i..].starts_with(pattern_bytes) {
            positions.push(i);
        }
    }
    
    Ok(positions)
}

/// Operaciones de hash rápidas
#[pyfunction]
fn fast_hash(data: &str) -> PyResult<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    Ok(hasher.finish())
}

/// Procesamiento de arrays numéricos
#[pyfunction]
fn fast_array_sum(numbers: Vec<f64>) -> PyResult<f64> {
    Ok(numbers.iter().sum())
}

#[pyfunction]
fn fast_array_max(numbers: Vec<f64>) -> PyResult<f64> {
    Ok(numbers.iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max))
}

#[pyfunction]
fn fast_array_min(numbers: Vec<f64>) -> PyResult<f64> {
    Ok(numbers.iter()
        .copied()
        .fold(f64::INFINITY, f64::min))
}

/// Media de array
#[pyfunction]
fn fast_array_mean(numbers: Vec<f64>) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot compute mean of empty array"
        ));
    }
    Ok(numbers.iter().sum::<f64>() / numbers.len() as f64)
}

/// Desviación estándar
#[pyfunction]
fn fast_array_std(numbers: Vec<f64>) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot compute std of empty array"
        ));
    }
    
    let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
    let variance = numbers.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / numbers.len() as f64;
    
    Ok(variance.sqrt())
}

/// Búsqueda binaria en array ordenado
#[pyfunction]
fn fast_binary_search(numbers: Vec<f64>, target: f64) -> PyResult<Option<usize>> {
    let mut left = 0;
    let mut right = numbers.len();
    
    while left < right {
        let mid = (left + right) / 2;
        if numbers[mid] < target {
            left = mid + 1;
        } else if numbers[mid] > target {
            right = mid;
        } else {
            return Ok(Some(mid));
        }
    }
    
    Ok(None)
}

/// Ordenar array rápidamente
#[pyfunction]
fn fast_array_sort(mut numbers: Vec<f64>) -> PyResult<Vec<f64>> {
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(numbers)
}

/// Contar ocurrencias de patrón en string
#[pyfunction]
fn fast_string_count(text: &str, pattern: &str) -> PyResult<usize> {
    if pattern.is_empty() {
        return Ok(0);
    }
    
    let mut count = 0;
    let pattern_bytes = pattern.as_bytes();
    let text_bytes = text.as_bytes();
    
    for i in 0..=text_bytes.len().saturating_sub(pattern_bytes.len()) {
        if text_bytes[i..].starts_with(pattern_bytes) {
            count += 1;
        }
    }
    
    Ok(count)
}

/// Reemplazar todas las ocurrencias en string
#[pyfunction]
fn fast_string_replace(text: &str, pattern: &str, replacement: &str) -> PyResult<String> {
    Ok(text.replace(pattern, replacement))
}

/// Validar formato JSON
#[pyfunction]
fn fast_json_validate(json_str: &str) -> PyResult<bool> {
    match serde_json::from_str::<serde_json::Value>(json_str) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Serializar a JSON rápidamente
#[pyfunction]
fn fast_json_stringify(value: PyObject) -> PyResult<String> {
    // Convertir objeto Python a JSON string
    // Nota: Implementación simplificada, se puede mejorar
    Python::with_gil(|py| {
        let json_module = py.import("json")?;
        let dumps = json_module.getattr("dumps")?;
        let result = dumps.call1((value,))?;
        Ok(result.to_string())
    })
}

/// Mediana de array
#[pyfunction]
fn fast_array_median(mut numbers: Vec<f64>) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot compute median of empty array"
        ));
    }
    
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = numbers.len() / 2;
    
    if numbers.len() % 2 == 0 {
        Ok((numbers[mid - 1] + numbers[mid]) / 2.0)
    } else {
        Ok(numbers[mid])
    }
}

/// Percentil de array
#[pyfunction]
fn fast_array_percentile(mut numbers: Vec<f64>, percentile: f64) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot compute percentile of empty array"
        ));
    }
    
    if percentile < 0.0 || percentile > 100.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Percentile must be between 0 and 100"
        ));
    }
    
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = (percentile / 100.0 * (numbers.len() - 1) as f64) as usize;
    Ok(numbers[index.min(numbers.len() - 1)])
}

/// Filtrar array por condición
#[pyfunction]
fn fast_array_filter(numbers: Vec<f64>, threshold: f64, op: &str) -> PyResult<Vec<f64>> {
    let filtered: Vec<f64> = match op {
        "gt" => numbers.into_iter().filter(|&x| x > threshold).collect(),
        "gte" => numbers.into_iter().filter(|&x| x >= threshold).collect(),
        "lt" => numbers.into_iter().filter(|&x| x < threshold).collect(),
        "lte" => numbers.into_iter().filter(|&x| x <= threshold).collect(),
        "eq" => numbers.into_iter().filter(|&x| (x - threshold).abs() < 1e-10).collect(),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Operation must be one of: gt, gte, lt, lte, eq"
        )),
    };
    Ok(filtered)
}

/// Split string por delimitador
#[pyfunction]
fn fast_string_split(text: &str, delimiter: &str) -> PyResult<Vec<String>> {
    Ok(text.split(delimiter).map(|s| s.to_string()).collect())
}

/// Join strings
#[pyfunction]
fn fast_string_join(strings: Vec<String>, separator: &str) -> PyResult<String> {
    Ok(strings.join(separator))
}

/// Trim whitespace
#[pyfunction]
fn fast_string_trim(text: &str) -> PyResult<String> {
    Ok(text.trim().to_string())
}

/// Convertir a mayúsculas
#[pyfunction]
fn fast_string_upper(text: &str) -> PyResult<String> {
    Ok(text.to_uppercase())
}

/// Convertir a minúsculas
#[pyfunction]
fn fast_string_lower(text: &str) -> PyResult<String> {
    Ok(text.to_lowercase())
}

/// Encontrar todas las ocurrencias de un patrón
#[pyfunction]
fn fast_string_find_all(text: &str, pattern: &str) -> PyResult<Vec<usize>> {
    let mut positions = Vec::new();
    let mut start = 0;
    
    while let Some(pos) = text[start..].find(pattern) {
        positions.push(start + pos);
        start += pos + 1;
    }
    
    Ok(positions)
}

/// Verificar si string empieza con patrón
#[pyfunction]
fn fast_string_starts_with(text: &str, pattern: &str) -> PyResult<bool> {
    Ok(text.starts_with(pattern))
}

/// Verificar si string termina con patrón
#[pyfunction]
fn fast_string_ends_with(text: &str, pattern: &str) -> PyResult<bool> {
    Ok(text.ends_with(pattern))
}

/// Calcular varianza de array
#[pyfunction]
fn fast_array_variance(numbers: Vec<f64>) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot compute variance of empty array"
        ));
    }
    
    let mean = numbers.iter().sum::<f64>() / numbers.len() as f64;
    let variance = numbers.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / numbers.len() as f64;
    
    Ok(variance)
}

/// Calcular rango (max - min) de array
#[pyfunction]
fn fast_array_range(numbers: Vec<f64>) -> PyResult<f64> {
    if numbers.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot compute range of empty array"
        ));
    }
    
    let min = numbers.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = numbers.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    Ok(max - min)
}

/// Calcular suma acumulativa
#[pyfunction]
fn fast_array_cumsum(numbers: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut cumsum = Vec::with_capacity(numbers.len());
    let mut sum = 0.0;
    
    for &num in &numbers {
        sum += num;
        cumsum.push(sum);
    }
    
    Ok(cumsum)
}

/// Calcular producto acumulativo
#[pyfunction]
fn fast_array_cumprod(numbers: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut cumprod = Vec::with_capacity(numbers.len());
    let mut prod = 1.0;
    
    for &num in &numbers {
        prod *= num;
        cumprod.push(prod);
    }
    
    Ok(cumprod)
}

/// Módulo Python
#[pymodule]
fn rust_extensions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_json_parse, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_search, m)?)?;
    m.add_function(wrap_pyfunction!(fast_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_sum, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_max, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_min, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_mean, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_std, m)?)?;
    m.add_function(wrap_pyfunction!(fast_binary_search, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_sort, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_count, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_replace, m)?)?;
    m.add_function(wrap_pyfunction!(fast_json_validate, m)?)?;
    m.add_function(wrap_pyfunction!(fast_json_stringify, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_median, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_filter, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_split, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_join, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_trim, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_upper, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_lower, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_find_all, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_starts_with, m)?)?;
    m.add_function(wrap_pyfunction!(fast_string_ends_with, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_variance, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_range, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(fast_array_cumprod, m)?)?;
    Ok(())
}

