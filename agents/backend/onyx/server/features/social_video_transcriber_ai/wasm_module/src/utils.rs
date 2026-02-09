//! Utility functions for WASM module

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hash_string(input: &str) -> String {
    let mut hash: u64 = 5381;
    for byte in input.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    format!("{:016x}", hash)
}

#[wasm_bindgen]
pub fn generate_id() -> String {
    let timestamp = js_sys::Date::now() as u64;
    let random = (js_sys::Math::random() * 1_000_000.0) as u64;
    format!("{:x}-{:x}", timestamp, random)
}

#[wasm_bindgen]
pub fn truncate_text(text: &str, max_length: usize, suffix: &str) -> String {
    if text.len() <= max_length {
        return text.to_string();
    }
    
    let truncate_at = max_length.saturating_sub(suffix.len());
    let boundary = text
        .char_indices()
        .take_while(|(i, _)| *i < truncate_at)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(truncate_at);
    
    format!("{}{}", &text[..boundary], suffix)
}

#[wasm_bindgen]
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[wasm_bindgen]
pub fn char_count(text: &str) -> usize {
    text.chars().count()
}

#[wasm_bindgen]
pub fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[wasm_bindgen]
pub fn to_title_case(text: &str) -> String {
    text.split_whitespace()
        .map(|word| capitalize(word))
        .collect::<Vec<String>>()
        .join(" ")
}

#[wasm_bindgen]
pub fn to_snake_case(text: &str) -> String {
    let mut result = String::with_capacity(text.len() + 5);
    for (i, c) in text.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.extend(c.to_lowercase());
    }
    result
}

#[wasm_bindgen]
pub fn to_camel_case(text: &str) -> String {
    let parts: Vec<&str> = text
        .split(|c: char| c == '_' || c == '-' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();

    if parts.is_empty() {
        return String::new();
    }

    let mut result = parts[0].to_lowercase();
    for part in parts.iter().skip(1) {
        result.push_str(&capitalize(part));
    }
    result
}

#[wasm_bindgen]
pub fn to_kebab_case(text: &str) -> String {
    to_snake_case(text).replace('_', "-")
}

#[wasm_bindgen]
pub fn escape_html(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            '\'' => "&#39;".to_string(),
            _ => c.to_string(),
        })
        .collect()
}

#[wasm_bindgen]
pub fn strip_html(text: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }
    
    result
}

#[wasm_bindgen]
pub fn is_valid_url(text: &str) -> bool {
    text.starts_with("http://") || text.starts_with("https://")
}

#[wasm_bindgen]
pub fn extract_domain(url: &str) -> Option<String> {
    url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .and_then(|s| s.split('/').next())
        .map(|s| s.to_string())
}

#[wasm_bindgen]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[wasm_bindgen]
pub fn format_duration(seconds: u64) -> String {
    match seconds {
        s if s < 60 => format!("{}s", s),
        s if s < 3600 => format!("{}m {}s", s / 60, s % 60),
        s if s < 86400 => format!("{}h {}m", s / 3600, (s % 3600) / 60),
        s => format!("{}d {}h", s / 86400, (s % 86400) / 3600),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate_text("hello world", 8, "..."), "hello...");
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
    }

    #[test]
    fn test_snake_case() {
        assert_eq!(to_snake_case("helloWorld"), "hello_world");
    }

    #[test]
    fn test_camel_case() {
        assert_eq!(to_camel_case("hello_world"), "helloWorld");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }
}












