/**
 * Storage Manager Module
 * ======================
 * Enhanced storage management with compression and versioning
 */

const StorageManager = {
    /**
     * Storage prefix
     */
    prefix: 'clothing_changer_',

    /**
     * Get item from storage
     */
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(this.prefix + key);
            if (item === null) {
                return defaultValue;
            }
            
            // Try to parse as JSON
            try {
                return JSON.parse(item);
            } catch {
                return item;
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Storage get error for key "${key}":`, error);
            }
            return defaultValue;
        }
    },

    /**
     * Set item in storage
     */
    set(key, value) {
        try {
            const serialized = typeof value === 'string' ? value : JSON.stringify(value);
            localStorage.setItem(this.prefix + key, serialized);
            
            // Emit storage event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('storage:set', { key, value });
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Storage set error for key "${key}":`, error);
            }
            
            // Handle quota exceeded
            if (error.name === 'QuotaExceededError') {
                this.clearOldData();
                // Try again
                try {
                    localStorage.setItem(this.prefix + key, JSON.stringify(value));
                    return true;
                } catch (retryError) {
                    if (typeof Notifications !== 'undefined') {
                        Notifications.error('No hay suficiente espacio de almacenamiento');
                    }
                }
            }
            
            return false;
        }
    },

    /**
     * Remove item from storage
     */
    remove(key) {
        try {
            localStorage.removeItem(this.prefix + key);
            
            // Emit storage event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('storage:remove', { key });
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Storage remove error for key "${key}":`, error);
            }
            return false;
        }
    },

    /**
     * Clear all items with prefix
     */
    clear() {
        try {
            const keys = Object.keys(localStorage);
            keys.forEach(key => {
                if (key.startsWith(this.prefix)) {
                    localStorage.removeItem(key);
                }
            });
            
            // Emit storage event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('storage:clear');
            }
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Storage clear error:', error);
            }
            return false;
        }
    },

    /**
     * Get all keys with prefix
     */
    keys() {
        const allKeys = Object.keys(localStorage);
        return allKeys
            .filter(key => key.startsWith(this.prefix))
            .map(key => key.substring(this.prefix.length));
    },

    /**
     * Get storage size estimate
     */
    getSize() {
        let total = 0;
        const keys = this.keys();
        
        keys.forEach(key => {
            const item = localStorage.getItem(this.prefix + key);
            if (item) {
                total += item.length + key.length;
            }
        });
        
        return {
            bytes: total,
            kb: (total / 1024).toFixed(2),
            mb: (total / 1024 / 1024).toFixed(2)
        };
    },

    /**
     * Clear old data when storage is full
     */
    clearOldData() {
        // Clear cache first
        if (typeof Cache !== 'undefined') {
            Cache.clear();
        }
        
        // Clear old history (keep last 20)
        const history = this.get('history', []);
        if (history.length > 20) {
            this.set('history', history.slice(0, 20));
        }
        
        // Clear old gallery (keep last 30)
        const gallery = this.get('gallery', []);
        if (gallery.length > 30) {
            this.set('gallery', gallery.slice(0, 30));
        }
    },

    /**
     * Export all data
     */
    export() {
        const data = {};
        const keys = this.keys();
        
        keys.forEach(key => {
            data[key] = this.get(key);
        });
        
        return JSON.stringify(data, null, 2);
    },

    /**
     * Import data
     */
    import(dataJson) {
        try {
            const data = JSON.parse(dataJson);
            
            Object.keys(data).forEach(key => {
                this.set(key, data[key]);
            });
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Storage import error:', error);
            }
            return false;
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StorageManager;
}

