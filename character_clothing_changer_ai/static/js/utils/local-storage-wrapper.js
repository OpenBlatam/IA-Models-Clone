/**
 * LocalStorage Wrapper Module
 * ============================
 * Generic wrapper for localStorage operations with error handling
 */

const LocalStorageWrapper = {
    /**
     * Get item from localStorage
     */
    get(key, defaultValue = null) {
        try {
            const value = localStorage.getItem(key);
            if (value === null) return defaultValue;
            
            // Try to parse as JSON
            try {
                return JSON.parse(value);
            } catch {
                // Not JSON, return as string
                return value;
            }
        } catch (e) {
            console.error(`Error loading from localStorage (${key}):`, e);
            return defaultValue;
        }
    },

    /**
     * Set item in localStorage
     */
    set(key, value) {
        try {
            const serialized = typeof value === 'string' ? value : JSON.stringify(value);
            localStorage.setItem(key, serialized);
            return true;
        } catch (e) {
            console.error(`Error saving to localStorage (${key}):`, e);
            // Handle quota exceeded
            if (e.name === 'QuotaExceededError') {
                console.warn('LocalStorage quota exceeded, attempting cleanup...');
                this.clearOldItems();
                // Retry once
                try {
                    const serialized = typeof value === 'string' ? value : JSON.stringify(value);
                    localStorage.setItem(key, serialized);
                    return true;
                } catch (e2) {
                    console.error('Retry failed:', e2);
                }
            }
            return false;
        }
    },

    /**
     * Remove item from localStorage
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error(`Error removing from localStorage (${key}):`, e);
            return false;
        }
    },

    /**
     * Clear all items
     */
    clear() {
        try {
            localStorage.clear();
            return true;
        } catch (e) {
            console.error('Error clearing localStorage:', e);
            return false;
        }
    },

    /**
     * Check if key exists
     */
    has(key) {
        return localStorage.getItem(key) !== null;
    },

    /**
     * Get all keys
     */
    keys() {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            keys.push(localStorage.key(i));
        }
        return keys;
    },

    /**
     * Clear old items to free space
     */
    clearOldItems() {
        // This is a placeholder - could implement LRU or size-based cleanup
        console.warn('clearOldItems not implemented');
    },

    /**
     * Get storage size estimate
     */
    getSizeEstimate() {
        let total = 0;
        for (let key in localStorage) {
            if (localStorage.hasOwnProperty(key)) {
                    total += localStorage[key].length + key.length;
            }
        }
        return total;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LocalStorageWrapper;
}

