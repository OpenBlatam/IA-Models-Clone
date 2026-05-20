/**
 * Storage Module
 * ==============
 * Handles localStorage operations using LocalStorageWrapper
 */

const Storage = {
    /**
     * Get history from localStorage
     */
    getHistory() {
        return LocalStorageWrapper.get(CONFIG.STORAGE_KEYS.HISTORY, []);
    },

    /**
     * Save history to localStorage
     */
    saveHistory(history) {
        return LocalStorageWrapper.set(CONFIG.STORAGE_KEYS.HISTORY, history);
    },

    /**
     * Get gallery from localStorage
     */
    getGallery() {
        return LocalStorageWrapper.get(CONFIG.STORAGE_KEYS.GALLERY, []);
    },

    /**
     * Save gallery to localStorage
     */
    saveGallery(gallery) {
        return LocalStorageWrapper.set(CONFIG.STORAGE_KEYS.GALLERY, gallery);
    },

    /**
     * Get theme from localStorage
     */
    getTheme() {
        return LocalStorageWrapper.get(CONFIG.STORAGE_KEYS.THEME, null);
    },

    /**
     * Save theme to localStorage
     */
    saveTheme(theme) {
        return LocalStorageWrapper.set(CONFIG.STORAGE_KEYS.THEME, theme);
    },

    /**
     * Get generic item
     */
    get(key, defaultValue = null) {
        return LocalStorageWrapper.get(key, defaultValue);
    },

    /**
     * Set generic item
     */
    set(key, value) {
        return LocalStorageWrapper.set(key, value);
    },

    /**
     * Remove item
     */
    remove(key) {
        return LocalStorageWrapper.remove(key);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Storage;
}

