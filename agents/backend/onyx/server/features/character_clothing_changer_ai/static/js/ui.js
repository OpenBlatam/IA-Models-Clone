/**
 * UI Module
 * ========
 * Handles UI interactions and updates
 */

const UI = {
    /**
     * Update status indicator
     */
    updateStatus(status, text) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        statusDot.className = `status-dot ${status}`;
        statusText.textContent = text;
    },

    /**
     * Toggle advanced options
     */
    toggleAdvanced() {
        const options = document.getElementById('advancedOptions');
        options.classList.toggle('show');
    },

    /**
     * Switch between tabs
     */
    switchTab(tabName) {
        // Track tab switch
        if (typeof AdvancedAnalytics !== 'undefined') {
            AdvancedAnalytics.trackUserAction('tab_switch', { tab: tabName });
        }
        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
        
        // Add active class to selected tab
        const tabs = document.querySelectorAll('.tab');
        const tabLabels = {
            'result': 'Resultado',
            'comparison': 'Comparación',
            'gallery': 'Galería',
            'history': 'Historial',
            'stats': 'Estadísticas'
        };
        
        tabs.forEach(tab => {
            if (tab.textContent.includes(tabLabels[tabName] || '')) {
                tab.classList.add('active');
            }
        });
        
        const tabElement = document.getElementById(tabName + 'Tab');
        if (tabElement) {
            tabElement.classList.add('active');
        }
        
        // Emit tab changed event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('tab:changed', tabName);
        }
        
        // Load stats if switching to stats tab
        if (tabName === 'stats') {
            const statsContent = document.getElementById('statsContent');
            if (statsContent && typeof Stats !== 'undefined') {
                statsContent.innerHTML = Stats.display();
            }
        }
    },

    /**
     * Show loading state
     */
    showLoading(message = 'Procesando...') {
        if (typeof ResultRenderer !== 'undefined') {
            return ResultRenderer.renderLoading(message);
        }
        // Fallback
        return `<div class="loading"><div class="spinner"></div><p>${message}</p></div>`;
    },

    /**
     * Show error message
     */
    showError(message) {
        if (typeof ResultRenderer !== 'undefined') {
            return ResultRenderer.renderError(message);
        }
        // Fallback
        return `<div class="error-message"><h3>❌ Error</h3><p><strong>${message}</strong></p></div>`;
    },

    /**
     * Show result
     */
    showResult(data) {
        if (typeof ResultRenderer !== 'undefined') {
            return ResultRenderer.renderResult(data);
        }
        // Fallback
        return `<div class="result-info"><h3>✅ Procesamiento Completado</h3></div>`;
    },

    /**
     * Toggle theme menu
     */
    toggleThemeMenu() {
        if (typeof ThemeManager !== 'undefined') {
            ThemeManager.toggleMenu();
        }
    },

    /**
     * Set theme
     */
    setTheme(theme) {
        if (typeof ThemeManager !== 'undefined') {
            ThemeManager.setTheme(theme);
        } else {
            // Fallback
            document.body.className = `theme-${theme}`;
            if (typeof Storage !== 'undefined') {
                Storage.saveTheme(theme);
            }
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UI;
}
