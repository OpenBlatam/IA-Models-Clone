/**
 * Translation Manager Module
 * ==========================
 * Advanced translation management with pluralization and interpolation
 */

const TranslationManager = {
    /**
     * Translations
     */
    translations: {},
    
    /**
     * Current language
     */
    currentLanguage: 'es',
    
    /**
     * Fallback language
     */
    fallbackLanguage: 'es',
    
    /**
     * Initialize translation manager
     */
    init() {
        // Load language from storage or browser
        const savedLanguage = localStorage.getItem('app_language');
        const browserLanguage = navigator.language.split('-')[0];
        
        this.currentLanguage = savedLanguage || browserLanguage || this.fallbackLanguage;
        
        // Load translations
        this.loadTranslations();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Translation manager initialized', { language: this.currentLanguage });
        }
    },
    
    /**
     * Set language
     */
    setLanguage(language) {
        this.currentLanguage = language;
        localStorage.setItem('app_language', language);
        
        // Reload translations
        this.loadTranslations();
        
        // Emit language change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('translation:language_changed', { language });
        }
        
        // Update HTML lang attribute
        document.documentElement.lang = language;
    },
    
    /**
     * Load translations
     */
    loadTranslations() {
        // Load from I18n if available
        if (typeof I18n !== 'undefined') {
            this.translations = I18n.translations || {};
        } else {
            // Fallback to default translations
            this.translations = this.getDefaultTranslations();
        }
    },
    
    /**
     * Get default translations
     */
    getDefaultTranslations() {
        return {
            es: {
                'common.loading': 'Cargando...',
                'common.error': 'Error',
                'common.success': 'Éxito',
                'form.submit': 'Cambiar Ropa',
                'form.processing': 'Procesando...'
            },
            en: {
                'common.loading': 'Loading...',
                'common.error': 'Error',
                'common.success': 'Success',
                'form.submit': 'Change Clothing',
                'form.processing': 'Processing...'
            }
        };
    },
    
    /**
     * Translate key
     */
    t(key, params = {}) {
        const translation = this.translations[key] || key;
        
        // Interpolate parameters
        let result = translation;
        Object.keys(params).forEach(param => {
            result = result.replace(new RegExp(`{${param}}`, 'g'), params[param]);
        });
        
        return result;
    },
    
    /**
     * Pluralize translation
     */
    plural(key, count, params = {}) {
        const pluralKey = count === 1 ? `${key}.singular` : `${key}.plural`;
        return this.t(pluralKey, { count, ...params });
    },
    
    /**
     * Add translations
     */
    addTranslations(language, translations) {
        if (!this.translations[language]) {
            this.translations[language] = {};
        }
        
        this.translations[language] = {
            ...this.translations[language],
            ...translations
        };
    },
    
    /**
     * Get current language
     */
    getCurrentLanguage() {
        return this.currentLanguage;
    },
    
    /**
     * Get available languages
     */
    getAvailableLanguages() {
        return Object.keys(this.translations);
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    TranslationManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TranslationManager;
}

