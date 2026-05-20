/**
 * Internationalization (i18n) Module
 * ==================================
 * Manages translations and localization
 */

const I18n = {
    /**
     * Current language
     */
    currentLanguage: 'es',
    
    /**
     * Translations storage
     */
    translations: {},
    
    /**
     * Default translations (Spanish)
     */
    defaultTranslations: {
        es: {
            // Common
            'common.loading': 'Cargando...',
            'common.error': 'Error',
            'common.success': 'Éxito',
            'common.cancel': 'Cancelar',
            'common.save': 'Guardar',
            'common.delete': 'Eliminar',
            'common.edit': 'Editar',
            'common.close': 'Cerrar',
            
            // Form
            'form.image.required': 'Por favor selecciona una imagen',
            'form.description.required': 'Debes describir la nueva ropa',
            'form.description.min': 'La descripción debe tener al menos {min} caracteres',
            'form.description.max': 'La descripción no puede exceder {max} caracteres',
            'form.submit': 'Cambiar Ropa',
            'form.processing': 'Procesando...',
            
            // Status
            'status.connecting': 'Conectando...',
            'status.connected': 'Conectado',
            'status.processing': 'Procesando...',
            'status.ready': 'Listo',
            'status.error': 'Error',
            'status.offline': 'Sin conexión',
            
            // Messages
            'message.clothing_changed': '¡Ropa cambiada exitosamente!',
            'message.connection_restored': 'Conexión restaurada',
            'message.connection_lost': 'Sin conexión. Los cambios se guardarán localmente.',
            'message.error_processing': 'Error al procesar la solicitud',
            
            // Tabs
            'tab.result': 'Resultado',
            'tab.comparison': 'Comparación',
            'tab.gallery': 'Galería',
            'tab.history': 'Historial',
            'tab.stats': 'Estadísticas',
            
            // Gallery
            'gallery.empty': 'No hay imágenes en la galería aún.',
            'gallery.search': 'Buscar en galería...',
            
            // History
            'history.empty': 'No hay historial aún.',
            'history.search': 'Buscar en historial...',
            
            // Errors
            'error.network': 'Error de conexión. Verifica que el servidor esté corriendo.',
            'error.server': 'Error del servidor. Por favor intenta nuevamente.',
            'error.validation': 'Error de validación. Por favor revisa los datos ingresados.',
            'error.unknown': 'Error desconocido. Por favor intenta nuevamente.'
        },
        en: {
            // Common
            'common.loading': 'Loading...',
            'common.error': 'Error',
            'common.success': 'Success',
            'common.cancel': 'Cancel',
            'common.save': 'Save',
            'common.delete': 'Delete',
            'common.edit': 'Edit',
            'common.close': 'Close',
            
            // Form
            'form.image.required': 'Please select an image',
            'form.description.required': 'You must describe the new clothing',
            'form.description.min': 'Description must be at least {min} characters',
            'form.description.max': 'Description cannot exceed {max} characters',
            'form.submit': 'Change Clothing',
            'form.processing': 'Processing...',
            
            // Status
            'status.connecting': 'Connecting...',
            'status.connected': 'Connected',
            'status.processing': 'Processing...',
            'status.ready': 'Ready',
            'status.error': 'Error',
            'status.offline': 'Offline',
            
            // Messages
            'message.clothing_changed': 'Clothing changed successfully!',
            'message.connection_restored': 'Connection restored',
            'message.connection_lost': 'No connection. Changes will be saved locally.',
            'message.error_processing': 'Error processing request',
            
            // Tabs
            'tab.result': 'Result',
            'tab.comparison': 'Comparison',
            'tab.gallery': 'Gallery',
            'tab.history': 'History',
            'tab.stats': 'Statistics',
            
            // Gallery
            'gallery.empty': 'No images in gallery yet.',
            'gallery.search': 'Search gallery...',
            
            // History
            'history.empty': 'No history yet.',
            'history.search': 'Search history...',
            
            // Errors
            'error.network': 'Connection error. Verify that the server is running.',
            'error.server': 'Server error. Please try again.',
            'error.validation': 'Validation error. Please check the entered data.',
            'error.unknown': 'Unknown error. Please try again.'
        }
    },
    
    /**
     * Initialize i18n
     */
    init(language = null) {
        // Detect language from browser or storage
        if (!language) {
            language = localStorage.getItem('app_language') || 
                      navigator.language.split('-')[0] || 
                      'es';
        }
        
        this.setLanguage(language);
        
        // Load translations
        this.loadTranslations();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('I18n initialized', { language: this.currentLanguage });
        }
    },
    
    /**
     * Set language
     */
    setLanguage(language) {
        if (!this.defaultTranslations[language]) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Language ${language} not supported, falling back to 'es'`);
            }
            language = 'es';
        }
        
        this.currentLanguage = language;
        localStorage.setItem('app_language', language);
        
        // Emit language change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('i18n:language_changed', { language });
        }
        
        // Update UI
        document.documentElement.lang = language;
    },
    
    /**
     * Load translations
     */
    loadTranslations() {
        this.translations = { ...this.defaultTranslations[this.currentLanguage] };
        
        // Load custom translations if available
        try {
            const custom = localStorage.getItem(`translations_${this.currentLanguage}`);
            if (custom) {
                const customTranslations = JSON.parse(custom);
                this.translations = { ...this.translations, ...customTranslations };
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to load custom translations', error);
            }
        }
    },
    
    /**
     * Translate a key
     */
    t(key, params = {}) {
        let translation = this.translations[key] || key;
        
        // Replace parameters
        Object.keys(params).forEach(param => {
            translation = translation.replace(`{${param}}`, params[param]);
        });
        
        return translation;
    },
    
    /**
     * Add custom translations
     */
    addTranslations(language, translations) {
        if (!this.defaultTranslations[language]) {
            this.defaultTranslations[language] = {};
        }
        
        this.defaultTranslations[language] = {
            ...this.defaultTranslations[language],
            ...translations
        };
        
        // Reload if current language
        if (language === this.currentLanguage) {
            this.loadTranslations();
        }
    },
    
    /**
     * Get available languages
     */
    getAvailableLanguages() {
        return Object.keys(this.defaultTranslations);
    },
    
    /**
     * Get current language
     */
    getCurrentLanguage() {
        return this.currentLanguage;
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    I18n.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = I18n;
}

