/**
 * Constants Module
 * ================
 * Centralized constants for the frontend application
 */

const CONSTANTS = {
    // API Configuration
    API: {
        BASE_URL: 'http://localhost:8002/api/v1',
        ENDPOINTS: {
            CHANGE_CLOTHING: '/change-clothing',
            HEALTH: '/health',
            MODEL_INFO: '/model/info',
            INITIALIZE: '/initialize',
            TENSORS: '/tensors',
            TENSOR: '/tensor',
            IMAGE: '/image'
        },
        TIMEOUT: 600000, // 10 minutes
        RETRY_ATTEMPTS: 3,
        RETRY_DELAY: 1000 // 1 second
    },
    
    // Storage Keys
    STORAGE: {
        HISTORY: 'clothingHistory',
        GALLERY: 'galleryItems',
        THEME: 'selectedTheme',
        FAVORITES: 'favorites',
        SETTINGS: 'appSettings'
    },
    
    // Limits
    LIMITS: {
        MAX_HISTORY: 50,
        MAX_GALLERY: 100,
        MAX_FAVORITES: 50,
        MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
        MIN_DESCRIPTION_LENGTH: 3,
        MAX_DESCRIPTION_LENGTH: 500,
        MIN_NAME_LENGTH: 1,
        MAX_NAME_LENGTH: 100
    },
    
    // Default Values
    DEFAULTS: {
        NUM_STEPS: 50,
        GUIDANCE_SCALE: 7.5,
        STRENGTH: 0.8,
        SAVE_TENSOR: true,
        THEME: 'default'
    },
    
    // Image Configuration
    IMAGE: {
        SUPPORTED_FORMATS: ['image/png', 'image/jpeg', 'image/jpg'],
        MAX_SIZE: 10 * 1024 * 1024, // 10MB
        PREVIEW_MAX_WIDTH: 800,
        PREVIEW_MAX_HEIGHT: 600
    },
    
    // Validation Messages
    VALIDATION: {
        IMAGE_REQUIRED: 'Por favor selecciona una imagen',
        IMAGE_INVALID_FORMAT: 'El archivo debe ser una imagen (PNG, JPG, JPEG)',
        IMAGE_TOO_LARGE: 'La imagen es demasiado grande (máximo 10MB)',
        DESCRIPTION_REQUIRED: 'Debes describir la nueva ropa',
        DESCRIPTION_TOO_SHORT: `La descripción debe tener al menos ${this.LIMITS.MIN_DESCRIPTION_LENGTH} caracteres`,
        DESCRIPTION_TOO_LONG: `La descripción no puede exceder ${this.LIMITS.MAX_DESCRIPTION_LENGTH} caracteres`,
        NAME_TOO_LONG: `El nombre no puede exceder ${this.LIMITS.MAX_NAME_LENGTH} caracteres`
    },
    
    // Error Messages
    ERRORS: {
        NETWORK_ERROR: 'Error de conexión. Verifica que el servidor esté corriendo.',
        SERVER_ERROR: 'Error del servidor. Por favor intenta nuevamente.',
        VALIDATION_ERROR: 'Error de validación. Por favor revisa los datos ingresados.',
        UNKNOWN_ERROR: 'Error desconocido. Por favor intenta nuevamente.'
    },
    
    // Success Messages
    SUCCESS: {
        CLOTHING_CHANGED: '¡Ropa cambiada exitosamente!',
        IMAGE_UPLOADED: 'Imagen cargada correctamente',
        SETTINGS_SAVED: 'Configuración guardada',
        ITEM_FAVORITED: 'Agregado a favoritos',
        ITEM_UNFAVORITED: 'Removido de favoritos'
    },
    
    // Status Messages
    STATUS: {
        CONNECTING: 'Conectando...',
        CONNECTED: 'Conectado',
        PROCESSING: 'Procesando...',
        READY: 'Listo',
        ERROR: 'Error',
        INITIALIZING: 'Inicializando modelo...'
    },
    
    // Cache Configuration
    CACHE: {
        DEFAULT_TTL: 5 * 60 * 1000, // 5 minutes
        MAX_SIZE: 100,
        CLEANUP_INTERVAL: 60000 // 1 minute
    },
    
    // Logger Configuration
    LOGGER: {
        DEFAULT_LEVEL: window.location.hostname === 'localhost' ? 'DEBUG' : 'INFO',
        MAX_HISTORY: 100
    },
    
    // Theme Options
    THEMES: {
        DEFAULT: 'default',
        DARK: 'dark',
        PURPLE: 'purple',
        BLUE: 'blue'
    },
    
    // Animation Durations (ms)
    ANIMATION: {
        FADE_IN: 300,
        FADE_OUT: 300,
        SLIDE_UP: 300,
        SLIDE_DOWN: 300
    },
    
    // Polling Intervals (ms)
    POLLING: {
        SERVER_STATUS: 30000, // 30 seconds
        MODEL_STATUS: 10000, // 10 seconds
        PROGRESS: 1000 // 1 second
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONSTANTS;
}

