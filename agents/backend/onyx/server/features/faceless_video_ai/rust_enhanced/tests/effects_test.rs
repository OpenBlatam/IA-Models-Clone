#[cfg(test)]
mod tests {
    use super::*;
    use faceless_video_enhanced::effects::EffectsEngine;
    use faceless_video_enhanced::error::VideoError;

    #[test]
    fn test_effects_engine_creation() {
        let engine = EffectsEngine::new(None);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_ken_burns_validation() {
        let engine = EffectsEngine::new(None).unwrap();
        
        // Test invalid duration
        let result = engine.ken_burns("test.jpg", -1.0, 1.2, 0.1, 0.1, None);
        assert!(result.is_err());
        
        // Test invalid zoom
        let result = engine.ken_burns("test.jpg", 5.0, 0.5, 0.1, 0.1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_blur_validation() {
        let engine = EffectsEngine::new(None).unwrap();
        
        // Test negative radius
        let result = engine.blur("test.jpg", -1.0, None);
        assert!(result.is_err());
    }
}












