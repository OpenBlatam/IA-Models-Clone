#[cfg(test)]
mod tests {
    use super::*;
    use faceless_video_enhanced::utils::*;
    use std::path::PathBuf;

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-1, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn test_validate_zoom() {
        assert!(validate_zoom(1.0).is_ok());
        assert!(validate_zoom(2.0).is_ok());
        assert!(validate_zoom(0.5).is_err());
        assert!(validate_zoom(15.0).is_err());
    }

    #[test]
    fn test_validate_pan() {
        assert_eq!(validate_pan(0.5), 0.5);
        assert_eq!(validate_pan(-0.5), 0.0);
        assert_eq!(validate_pan(1.5), 1.0);
    }

    #[test]
    fn test_validate_duration() {
        assert!(validate_duration(1.0).is_ok());
        assert!(validate_duration(0.0).is_err());
        assert!(validate_duration(-1.0).is_err());
        assert!(validate_duration(4000.0).is_err());
    }

    #[test]
    fn test_generate_output_path() {
        let output_dir = PathBuf::from("/tmp");
        let path = generate_output_path("test.jpg", &output_dir, "processed", Some("png"));
        assert!(path.to_string_lossy().contains("processed"));
        assert!(path.to_string_lossy().ends_with(".png"));
    }
}












