/**
 * Storage Manager V2 Module
 * ========================
 * Advanced storage management with encryption and compression
 */

const StorageManagerV2 = {
    /**
     * Storage adapters
     */
    adapters: new Map(),
    
    /**
     * Current adapter
     */
    currentAdapter: 'localStorage',
    
    /**
     * Encryption key
     */
    encryptionKey: null,
    
    /**
     * Initialize storage manager
     */
    init() {
        // Register default adapters
        this.registerAdapter('localStorage', {
            get: (key) => {
                try {
                    const value = localStorage.getItem(key);
                    return value ? JSON.parse(value) : null;
                } catch (error) {
                    return null;
                }
            },
            set: (key, value) => {
                try {
                    localStorage.setItem(key, JSON.stringify(value));
                    return true;
                } catch (error) {
                    return false;
                }
            },
            remove: (key) => {
                try {
                    localStorage.removeItem(key);
                    return true;
                } catch (error) {
                    return false;
                }
            },
            clear: () => {
                try {
                    localStorage.clear();
                    return true;
                } catch (error) {
                    return false;
                }
            },
            keys: () => {
                return Object.keys(localStorage);
            }
        });
        
        this.registerAdapter('sessionStorage', {
            get: (key) => {
                try {
                    const value = sessionStorage.getItem(key);
                    return value ? JSON.parse(value) : null;
                } catch (error) {
                    return null;
                }
            },
            set: (key, value) => {
                try {
                    sessionStorage.setItem(key, JSON.stringify(value));
                    return true;
                } catch (error) {
                    return false;
                }
            },
            remove: (key) => {
                try {
                    sessionStorage.removeItem(key);
                    return true;
                } catch (error) {
                    return false;
                }
            },
            clear: () => {
                try {
                    sessionStorage.clear();
                    return true;
                } catch (error) {
                    return false;
                }
            },
            keys: () => {
                return Object.keys(sessionStorage);
            }
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Storage manager V2 initialized');
        }
    },
    
    /**
     * Register adapter
     */
    registerAdapter(name, adapter) {
        this.adapters.set(name, adapter);
    },
    
    /**
     * Set adapter
     */
    setAdapter(name) {
        if (this.adapters.has(name)) {
            this.currentAdapter = name;
            return true;
        }
        return false;
    },
    
    /**
     * Get value
     */
    get(key, defaultValue = null) {
        const adapter = this.adapters.get(this.currentAdapter);
        if (!adapter) {
            return defaultValue;
        }
        
        const value = adapter.get(key);
        return value !== null ? value : defaultValue;
    },
    
    /**
     * Set value
     */
    set(key, value, options = {}) {
        const {
            encrypt = false,
            compress = false,
            ttl = null
        } = options;
        
        let processedValue = value;
        
        // Encrypt if needed
        if (encrypt && this.encryptionKey) {
            processedValue = this.encrypt(processedValue);
        }
        
        // Compress if needed
        if (compress) {
            processedValue = this.compress(processedValue);
        }
        
        // Add TTL metadata
        if (ttl) {
            processedValue = {
                value: processedValue,
                expires: Date.now() + ttl
            };
        }
        
        const adapter = this.adapters.get(this.currentAdapter);
        if (!adapter) {
            return false;
        }
        
        return adapter.set(key, processedValue);
    },
    
    /**
     * Remove value
     */
    remove(key) {
        const adapter = this.adapters.get(this.currentAdapter);
        if (!adapter) {
            return false;
        }
        return adapter.remove(key);
    },
    
    /**
     * Clear all
     */
    clear() {
        const adapter = this.adapters.get(this.currentAdapter);
        if (!adapter) {
            return false;
        }
        return adapter.clear();
    },
    
    /**
     * Get all keys
     */
    keys() {
        const adapter = this.adapters.get(this.currentAdapter);
        if (!adapter) {
            return [];
        }
        return adapter.keys();
    },
    
    /**
     * Check if key exists
     */
    has(key) {
        return this.get(key) !== null;
    },
    
    /**
     * Get size
     */
    getSize(key) {
        const value = this.get(key);
        if (!value) {
            return 0;
        }
        return new Blob([JSON.stringify(value)]).size;
    },
    
    /**
     * Get all data
     */
    getAll() {
        const keys = this.keys();
        const data = {};
        keys.forEach(key => {
            data[key] = this.get(key);
        });
        return data;
    },
    
    /**
     * Encrypt (simple base64 encoding for demo)
     */
    encrypt(data) {
        try {
            const json = JSON.stringify(data);
            return btoa(json);
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Encryption error', error);
            }
            return data;
        }
    },
    
    /**
     * Decrypt
     */
    decrypt(encryptedData) {
        try {
            const json = atob(encryptedData);
            return JSON.parse(json);
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Decryption error', error);
            }
            return encryptedData;
        }
    },
    
    /**
     * Compress (simple JSON stringify for demo)
     */
    compress(data) {
        return JSON.stringify(data);
    },
    
    /**
     * Decompress
     */
    decompress(compressedData) {
        return JSON.parse(compressedData);
    },
    
    /**
     * Clean expired items
     */
    cleanExpired() {
        const keys = this.keys();
        let cleaned = 0;
        
        keys.forEach(key => {
            const value = this.get(key);
            if (value && value.expires && Date.now() > value.expires) {
                this.remove(key);
                cleaned++;
            }
        });
        
        return cleaned;
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    StorageManagerV2.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StorageManagerV2;
}

