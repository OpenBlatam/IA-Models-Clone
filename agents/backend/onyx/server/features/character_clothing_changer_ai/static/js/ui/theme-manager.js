/**
 * Theme Manager Module
 * ====================
 * Handles theme switching and management
 */

const ThemeManager = {
    /**
     * Toggle theme menu visibility
     */
    toggleMenu() {
        const menu = document.getElementById('themeMenu');
        if (menu) {
            menu.classList.toggle('show');
        }
    },

    /**
     * Set theme
     */
    setTheme(theme) {
        document.body.className = `theme-${theme}`;
        
        // Save to storage
        if (typeof Storage !== 'undefined') {
            Storage.saveTheme(theme);
        }
        
        // Update state manager
        if (typeof StateManager !== 'undefined') {
            StateManager.set('theme', theme);
        }
        
        // Close menu
        this.toggleMenu();
        
        // Emit theme change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('theme:changed', theme);
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Theme changed', { theme });
        }
    },

    /**
     * Get current theme
     */
    getCurrentTheme() {
        if (typeof StateManager !== 'undefined') {
            return StateManager.get('theme');
        }
        if (typeof Storage !== 'undefined') {
            return Storage.getTheme();
        }
        return null;
    },

    /**
     * Setup theme menu click outside handler
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
     * Setup theme option click handlers
     */
    setupThemeOptions() {
        const themeOptions = document.querySelectorAll('.theme-option[data-theme]');
        themeOptions.forEach(option => {
            option.addEventListener('click', () => {
                const theme = option.getAttribute('data-theme');
                this.setTheme(theme);
            });
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeManager;
}

