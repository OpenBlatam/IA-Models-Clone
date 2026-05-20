/**
 * Internationalization (i18n) Module
 * ===================================
 * Multi-language support
 */

const I18n = {
    /**
     * Current language
     */
    currentLanguage: 'es',
    
    /**
     * Translations
     */
    translations: {
        es: {
            'app.title': 'Character Clothing Changer AI',
            'app.subtitle': 'Cambia la ropa de tus personajes con IA usando Flux2',
            'form.image.label': 'Imagen del Personaje',
            'form.clothing.label': 'Descripción de la Ropa',
            'form.submit': 'Cambiar Ropa',
            'form.processing': 'Procesando...',
            'notification.success': 'Operación completada exitosamente',
            'notification.error': 'Ha ocurrido un error',
            'notification.warning': 'Advertencia',
            'notification.info': 'Información',
            'gallery.empty': 'No hay imágenes en la galería aún.',
            'history.empty': 'No hay historial aún.',
            'search.placeholder': 'Buscar...',
            'stats.title': 'Estadísticas',
            'stats.processed': 'Procesamientos',
            'stats.gallery': 'En Galería',
            'stats.favorites': 'Favoritos'
        },
        en: {
            'app.title': 'Character Clothing Changer AI',
            'app.subtitle': 'Change your characters clothing with AI using Flux2',
            'form.image.label': 'Character Image',
            'form.clothing.label': 'Clothing Description',
            'form.submit': 'Change Clothing',
            'form.processing': 'Processing...',
            'notification.success': 'Operation completed successfully',
            'notification.error': 'An error occurred',
            'notification.warning': 'Warning',
            'notification.info': 'Information',
            'gallery.empty': 'No images in gallery yet.',
            'history.empty': 'No history yet.',
            'search.placeholder': 'Search...',
            'stats.title': 'Statistics',
            'stats.processed': 'Processed',
            'stats.gallery': 'In Gallery',
            'stats.favorites': 'Favorites'
        }
    },
    
    /**
     * Initialize i18n
     */
    init() {
        // Detect language from browser
        const browserLang = navigator.language || navigator.userLanguage;
        const lang = browserLang.split('-')[0];
        
        // Set language if available
        if (this.translations[lang]) {
            this.currentLanguage = lang;
        }
        
        // Load saved language preference
        if (typeof Storage !== 'undefined') {
            const savedLang = Storage.get('language');
            if (savedLang && this.translations[savedLang]) {
                this.currentLanguage = savedLang;
            }
        }
        
        // Apply translations
        this.applyTranslations();
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`i18n initialized with language: ${this.currentLanguage}`);
        }
    },
    
    /**
     * Set language
     */
    setLanguage(lang) {
        if (!this.translations[lang]) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Language not available: ${lang}`);
            }
            return false;
        }
        
        this.currentLanguage = lang;
        
        // Save preference
        if (typeof Storage !== 'undefined') {
            Storage.save('language', lang);
        }
        
        // Apply translations
        this.applyTranslations();
        
        // Emit language change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('i18n:language:changed', lang);
        }
        
        return true;
    },
    
    /**
     * Translate key
     */
    t(key, params = {}) {
        const translation = this.translations[this.currentLanguage]?.[key] || key;
        
        // Replace parameters
        return translation.replace(/\{\{(\w+)\}\}/g, (match, param) => {
            return params[param] !== undefined ? params[param] : match;
        });
    },
    
    /**
     * Apply translations to DOM
     */
    applyTranslations() {
        // Find all elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.t(key);
            
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.placeholder = translation;
            } else {
                element.textContent = translation;
            }
        });
    },
    
    /**
     * Add translations
     */
    addTranslations(lang, translations) {
        if (!this.translations[lang]) {
            this.translations[lang] = {};
        }
        
        this.translations[lang] = {
            ...this.translations[lang],
            ...translations
        };
    },
    
    /**
     * Get available languages
     */
    getAvailableLanguages() {
        return Object.keys(this.translations);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = I18n;
}

