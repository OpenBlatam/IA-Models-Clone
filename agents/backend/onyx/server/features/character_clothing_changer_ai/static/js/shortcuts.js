/**
 * Keyboard Shortcuts Module
 * =========================
 * Handles keyboard shortcuts
 */

const Shortcuts = {
    /**
     * Initialize keyboard shortcuts
     */
    init() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter: Submit form
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                const form = document.getElementById('clothingForm');
                if (form) {
                    form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
                }
            }

            // Ctrl/Cmd + K: Focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('.search-input');
                if (searchInput) {
                    searchInput.focus();
                }
            }

            // Escape: Close modals/menus
            if (e.key === 'Escape') {
                const themeMenu = document.getElementById('themeMenu');
                if (themeMenu && themeMenu.classList.contains('show')) {
                    themeMenu.classList.remove('show');
                }
            }

            // Number keys 1-4: Switch tabs
            if (e.key >= '1' && e.key <= '4' && !e.ctrlKey && !e.metaKey) {
                const tabs = ['result', 'comparison', 'gallery', 'history'];
                const tabIndex = parseInt(e.key) - 1;
                if (tabs[tabIndex]) {
                    UI.switchTab(tabs[tabIndex]);
                }
            }
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Shortcuts;
}


