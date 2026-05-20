/**
 * Config Manager Module
 * =====================
 * Dynamic configuration management
 */

const ConfigManager = {
    /**
     * Configuration store
     */
    config: {},
    
    /**
     * Configuration watchers
     */
    watchers: new Map(),
    
    /**
     * Initialize config manager
     */
    init(defaultConfig = {}) {
        // Load from localStorage
        if (typeof Storage !== 'undefined') {
            const saved = Storage.get('app_config');
            if (saved) {
                this.config = { ...defaultConfig, ...saved };
            } else {
                this.config = { ...defaultConfig };
            }
        } else {
            this.config = { ...defaultConfig };
        }
        
        // Merge with CONFIG if available
        if (typeof CONFIG !== 'undefined') {
            this.config = { ...CONFIG, ...this.config };
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Config manager initialized');
        }
    },
    
    /**
     * Get config value
     */
    get(key, defaultValue = null) {
        if (!key) {
            return { ...this.config };
        }
        
        const keys = key.split('.');
        let value = this.config;
        
        for (const k of keys) {
            if (value && typeof value === 'object' && k in value) {
                value = value[k];
            } else {
                return defaultValue;
            }
        }
        
        return value;
    },
    
    /**
     * Set config value
     */
    set(key, value) {
        const keys = key.split('.');
        let current = this.config;
        
        for (let i = 0; i < keys.length - 1; i++) {
            const k = keys[i];
            if (!(k in current) || typeof current[k] !== 'object') {
                current[k] = {};
            }
            current = current[k];
        }
        
        const lastKey = keys[keys.length - 1];
        const oldValue = current[lastKey];
        current[lastKey] = value;
        
        // Save to storage
        if (typeof Storage !== 'undefined') {
            Storage.save('app_config', this.config);
        }
        
        // Notify watchers
        this.notifyWatchers(key, value, oldValue);
        
        // Emit config change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('config:changed', { key, value, oldValue });
        }
        
        return true;
    },
    
    /**
     * Watch config changes
     */
    watch(key, callback) {
        if (!this.watchers.has(key)) {
            this.watchers.set(key, []);
        }
        
        this.watchers.get(key).push(callback);
        
        // Return unwatch function
        return () => {
            const watchers = this.watchers.get(key);
            if (watchers) {
                const index = watchers.indexOf(callback);
                if (index > -1) {
                    watchers.splice(index, 1);
                }
            }
        };
    },
    
    /**
     * Notify watchers
     */
    notifyWatchers(key, newValue, oldValue) {
        // Exact match
        const exactWatchers = this.watchers.get(key);
        if (exactWatchers) {
            exactWatchers.forEach(callback => {
                try {
                    callback(newValue, oldValue, key);
                } catch (error) {
                    if (typeof Logger !== 'undefined') {
                        Logger.error(`Error in config watcher for ${key}:`, error);
                    }
                }
            });
        }
        
        // Parent key watchers
        const parts = key.split('.');
        for (let i = parts.length - 1; i > 0; i--) {
            const parentKey = parts.slice(0, i).join('.');
            const parentWatchers = this.watchers.get(parentKey);
            if (parentWatchers) {
                parentWatchers.forEach(callback => {
                    try {
                        callback(newValue, oldValue, key);
                    } catch (error) {
                        if (typeof Logger !== 'undefined') {
                            Logger.error(`Error in config watcher for ${parentKey}:`, error);
                        }
                    }
                });
            }
        }
    },
    
    /**
     * Reset config to defaults
     */
    reset() {
        this.config = {};
        this.watchers.clear();
        
        if (typeof Storage !== 'undefined') {
            Storage.remove('app_config');
        }
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('config:reset');
        }
    },
    
    /**
     * Export config
     */
    export() {
        return JSON.stringify(this.config, null, 2);
    },
    
    /**
     * Import config
     */
    import(configJson) {
        try {
            const imported = JSON.parse(configJson);
            this.config = { ...this.config, ...imported };
            
            // Save to storage
            if (typeof Storage !== 'undefined') {
                Storage.save('app_config', this.config);
            }
            
            // Notify all watchers
            Object.keys(imported).forEach(key => {
                this.notifyWatchers(key, imported[key], this.config[key]);
            });
            
            return true;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Config import error:', error);
            }
            return false;
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConfigManager;
}

