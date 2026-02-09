/**
 * Configuration Module
 * ====================
 * Centralized configuration for the application
 * 
 * Note: This module uses CONSTANTS for actual values.
 * This is kept for backward compatibility.
 */

// Use CONSTANTS if available, otherwise use defaults
const CONFIG = typeof CONSTANTS !== 'undefined' ? {
    API_BASE: CONSTANTS.API.BASE_URL,
    STORAGE_KEYS: CONSTANTS.STORAGE,
    LIMITS: CONSTANTS.LIMITS,
    DEFAULT_VALUES: {
        NUM_STEPS: CONSTANTS.DEFAULTS.NUM_STEPS,
        GUIDANCE_SCALE: CONSTANTS.DEFAULTS.GUIDANCE_SCALE,
        STRENGTH: CONSTANTS.DEFAULTS.STRENGTH,
        SAVE_TENSOR: CONSTANTS.DEFAULTS.SAVE_TENSOR.toString()
    },
    ERROR_REPORTING: false, // Set to true to enable error reporting to server
    LOG_LEVEL: CONSTANTS.LOGGER.DEFAULT_LEVEL || 'INFO' // DEBUG, INFO, WARN, ERROR, NONE
} : {
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


