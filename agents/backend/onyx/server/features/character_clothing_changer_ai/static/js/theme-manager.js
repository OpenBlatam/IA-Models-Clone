/**
 * Theme Manager Module
 * ====================
 * Advanced theme management system
 */

const ThemeManager = {
    /**
     * Available themes
     */
    themes: {
        default: {
            name: 'Claro',
            icon: '🌙',
            colors: {
                primary: '#667eea',
                secondary: '#764ba2',
                background: '#ffffff',
                text: '#333333'
            }
        },
        dark: {
            name: 'Oscuro',
            icon: '🌚',
            colors: {
                primary: '#4a5568',
                secondary: '#2d3748',
                background: '#1a202c',
                text: '#e2e8f0'
            }
        },
        purple: {
            name: 'Púrpura',
            icon: '💜',
            colors: {
                primary: '#9f7aea',
                secondary: '#805ad5',
                background: '#f7fafc',
                text: '#2d3748'
            }
        },
        blue: {
            name: 'Azul',
            icon: '💙',
            colors: {
                primary: '#4299e1',
                secondary: '#3182ce',
                background: '#ebf8ff',
                text: '#2c5282'
            }
        }
    },
    
    /**
     * Current theme
     */
    currentTheme: 'default',
    
    /**
     * Initialize theme manager
     */
    init() {
        // Load saved theme
        if (typeof Storage !== 'undefined') {
            const saved = Storage.getTheme();
            if (saved && this.themes[saved]) {
                this.currentTheme = saved;
            }
        }
        
        // Apply theme
        this.applyTheme(this.currentTheme);
        
        // Setup theme menu
        this.setupThemeMenu();
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Theme manager initialized with theme: ${this.currentTheme}`);
        }
    },
    
    /**
     * Apply theme
     */
    applyTheme(themeName) {
        if (!this.themes[themeName]) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Theme not found: ${themeName}`);
            }
            return false;
        }
        
        this.currentTheme = themeName;
        const theme = this.themes[themeName];
        
        // Apply CSS variables
        const root = document.documentElement;
        Object.entries(theme.colors).forEach(([key, value]) => {
            root.style.setProperty(`--theme-${key}`, value);
        });
        
        // Apply body class
        document.body.className = `theme-${themeName}`;
        
        // Save theme
        if (typeof Storage !== 'undefined') {
            Storage.saveTheme(themeName);
        }
        
        // Update state
        if (typeof StateManager !== 'undefined') {
            StateManager.set('theme', themeName);
        }
        
        // Emit theme changed event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('theme:changed', { theme: themeName, themeData: theme });
        }
        
        return true;
    },
    
    /**
     * Get current theme
     */
    getCurrentTheme() {
        return {
            name: this.currentTheme,
            data: this.themes[this.currentTheme]
        };
    },
    
    /**
     * Get all themes
     */
    getAllThemes() {
        return Object.entries(this.themes).map(([key, theme]) => ({
            id: key,
            ...theme
        }));
    },
    
    /**
     * Setup theme menu
     */
    setupThemeMenu() {
        const themeMenu = document.getElementById('themeMenu');
        if (!themeMenu) return;
        
        // Clear existing options
        themeMenu.innerHTML = '';
        
        // Add theme options
        Object.entries(this.themes).forEach(([key, theme]) => {
            const option = document.createElement('div');
            option.className = 'theme-option';
            option.setAttribute('data-theme', key);
            option.innerHTML = `${theme.icon} ${theme.name}`;
            option.addEventListener('click', () => {
                this.applyTheme(key);
                this.toggleMenu();
            });
            themeMenu.appendChild(option);
        });
    },
    
    /**
     * Toggle theme menu
     */
    toggleMenu() {
        const themeMenu = document.getElementById('themeMenu');
        if (themeMenu) {
            themeMenu.classList.toggle('show');
        }
    },
    
    /**
     * Setup click outside handler
     */
    setupClickOutsideHandler() {
        document.addEventListener('click', (e) => {
            const themeMenu = document.getElementById('themeMenu');
            const themeBtn = document.querySelector('.theme-btn');
            
            if (themeMenu && themeBtn &&
                !themeMenu.contains(e.target) &&
                !themeBtn.contains(e.target)) {
                themeMenu.classList.remove('show');
            }
        });
    },
    
    /**
     * Setup theme options (for backward compatibility)
     */
    setupThemeOptions() {
        // Already handled in setupThemeMenu
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeManager;
}

