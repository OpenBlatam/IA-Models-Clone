//! String Extensions
//!
//! Extension trait for String and &str.

/// Extension trait for String.
pub trait StringExt {
    /// Check if string is not empty.
    fn is_not_empty(&self) -> bool;
    
    /// Truncate to max length with ellipsis.
    fn truncate_with_ellipsis(&self, max_len: usize) -> String;
    
    /// Remove whitespace from both ends.
    fn trim_whitespace(&self) -> String;
    
    /// Check if string contains only whitespace.
    fn is_whitespace_only(&self) -> bool;
}

impl StringExt for String {
    fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }
    
    fn truncate_with_ellipsis(&self, max_len: usize) -> String {
        if self.len() <= max_len {
            return self.clone();
        }
        format!("{}...", &self[..max_len.saturating_sub(3)])
    }
    
    fn trim_whitespace(&self) -> String {
        self.trim().to_string()
    }
    
    fn is_whitespace_only(&self) -> bool {
        self.trim().is_empty()
    }
}

impl StringExt for &str {
    fn is_not_empty(&self) -> bool {
        !self.is_empty()
    }
    
    fn truncate_with_ellipsis(&self, max_len: usize) -> String {
        if self.len() <= max_len {
            return self.to_string();
        }
        format!("{}...", &self[..max_len.saturating_sub(3)])
    }
    
    fn trim_whitespace(&self) -> String {
        self.trim().to_string()
    }
    
    fn is_whitespace_only(&self) -> bool {
        self.trim().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_ext() {
        let s = "  hello world  ";
        assert!(s.is_not_empty());
        assert_eq!(s.truncate_with_ellipsis(5), "he...");
        assert_eq!(s.trim_whitespace(), "hello world");
        assert!(!s.is_whitespace_only());
        assert!("   ".is_whitespace_only());
    }
}




