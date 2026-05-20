/**
 * Theme Engine Module
 * ===================
 * Advanced theme management with dynamic theming
 */

const ThemeEngine = {
    /**
     * Available themes
     */
    themes: new Map(),
    
    /**
     * Current theme
     */
    currentTheme: 'default',
    
    /**
     * Theme variables
     */
    variables: new Map(),
    
    /**
     * Initialize theme engine
     */
    init() {
        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'default';
        this.setTheme(savedTheme);
        
        // Register default themes
        this.registerDefaultThemes();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Theme engine initialized', { theme: this.currentTheme });
        }
    },
    
    /**
     * Register default themes
     */
    registerDefaultThemes() {
        this.registerTheme('default', {
            name: 'Default',
            colors: {
                primary: '#007bff',
                secondary: '#6c757d',
                success: '#28a745',
                danger: '#dc3545',
                warning: '#ffc107',
                info: '#17a2b8'
            }
        });
        
        this.registerTheme('dark', {
            name: 'Dark',
            colors: {
                primary: '#4a9eff',
                secondary: '#6c757d',
                success: '#4caf50',
                danger: '#f44336',
                warning: '#ff9800',
                info: '#00bcd4'
            }
        });
    },
    
    /**
     * Register theme
     */
    registerTheme(name, theme) {
        this.themes.set(name, theme);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Theme registered: ${name}`);
        }
    },
    
    /**
     * Set theme
     */
    setTheme(name) {
        const theme = this.themes.get(name);
        if (!theme) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Theme not found: ${name}`);
            }
            return false;
        }
        
        this.currentTheme = name;
        localStorage.setItem('theme', name);
        
        // Apply theme
        this.applyTheme(theme);
        
        // Emit theme change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('theme:changed', { name, theme });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Theme changed: ${name}`);
        }
        
        return true;
    },
    
    /**
     * Apply theme
     */
    applyTheme(theme) {
        const root = document.documentElement;
        
        // Remove old theme class
        root.classList.remove(`theme-${this.currentTheme}`);
        
        // Add new theme class
        root.classList.add(`theme-${theme.name.toLowerCase()}`);
        
        // Apply CSS variables
        if (theme.colors) {
            Object.keys(theme.colors).forEach(key => {
                root.style.setProperty(`--color-${key}`, theme.colors[key]);
            });
        }
        
        // Apply custom CSS if provided
        if (theme.css) {
            this.applyCustomCSS(theme.css);
        }
    },
    
    /**
     * Apply custom CSS
     */
    applyCustomCSS(css) {
        let styleElement = document.getElementById('theme-custom-css');
        if (!styleElement) {
            styleElement = document.createElement('style');
            styleElement.id = 'theme-custom-css';
            document.head.appendChild(styleElement);
        }
        styleElement.textContent = css;
    },
    
    /**
     * Get current theme
     */
    getCurrentTheme() {
        return this.currentTheme;
    },
    
    /**
     * Get theme
     */
    getTheme(name) {
        return this.themes.get(name);
    },
    
    /**
     * Get all themes
     */
    getAllThemes() {
        return Array.from(this.themes.values());
    },
    
    /**
     * Create custom theme
     */
    createCustomTheme(name, colors, css = null) {
        this.registerTheme(name, {
            name,
            colors,
            css
        });
    },
    
    /**
     * Update theme variable
     */
    updateVariable(key, value) {
        this.variables.set(key, value);
        document.documentElement.style.setProperty(`--${key}`, value);
    },
    
    /**
     * Get theme variable
     */
    getVariable(key) {
        return this.variables.get(key) || 
               getComputedStyle(document.documentElement).getPropertyValue(`--${key}`).trim();
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    ThemeEngine.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeEngine;
}

