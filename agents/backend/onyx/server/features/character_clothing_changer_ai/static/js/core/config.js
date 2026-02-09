/**
 * Configuration Module
 * ====================
 * Centralized configuration for the application
 */

const CONFIG = {
    API_BASE: 'http://localhost:8002/api/v1',
    STORAGE_KEYS: {
        HISTORY: 'clothingHistory',
        GALLERY: 'galleryItems',
        THEME: 'selectedTheme'
    },
    LIMITS: {
        MAX_HISTORY: 50,
        MAX_GALLERY: 100
    },
    DEFAULT_VALUES: {
        NUM_STEPS: 50,
        GUIDANCE_SCALE: 7.5,
        STRENGTH: 0.8,
        SAVE_TENSOR: 'true'
    },
    ERROR_REPORTING: false, // Set to true to enable error reporting to server
    LOG_LEVEL: 'INFO' // DEBUG, INFO, WARN, ERROR, NONE
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}

