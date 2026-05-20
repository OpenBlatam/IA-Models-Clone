/**
 * Cache Module
 * ============
 * Client-side caching system with TTL support
 */

const Cache = {
    /**
     * Cache storage
     */
    storage: new Map(),
    
    /**
     * Default TTL (Time To Live) in milliseconds
     */
    defaultTTL: 5 * 60 * 1000, // 5 minutes
    
    /**
     * Set a value in cache
     */
    set(key, value, ttl = null) {
        const expiry = ttl === null 
            ? Date.now() + this.defaultTTL 
            : Date.now() + ttl;
        
        this.storage.set(key, {
            value,
            expiry,
            createdAt: Date.now()
        });
        
        // Log cache set
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Cache set: ${key}`, { ttl: ttl || this.defaultTTL });
        }
        
        // Emit cache event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('cache:set', key, value);
        }
    },
    
    /**
     * Get a value from cache
     */
    get(key) {
        const item = this.storage.get(key);
        
        if (!item) {
            return null;
        }
        
        // Check if expired
        if (Date.now() > item.expiry) {
            this.delete(key);
            return null;
        }
        
        return item.value;
    },
    
    /**
     * Check if key exists and is not expired
     */
    has(key) {
        return this.get(key) !== null;
    },
    
    /**
     * Delete a key from cache
     */
    delete(key) {
        const deleted = this.storage.delete(key);
        
        if (deleted && typeof EventBus !== 'undefined') {
            EventBus.emit('cache:delete', key);
        }
        
        return deleted;
    },
    
    /**
     * Clear all cache
     */
    clear() {
        const size = this.storage.size;
        this.storage.clear();
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('cache:clear', size);
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Cache cleared: ${size} items removed`);
        }
    },
    
    /**
     * Clear expired items
     */
    clearExpired() {
        const now = Date.now();
        let cleared = 0;
        
        this.storage.forEach((item, key) => {
            if (now > item.expiry) {
                this.storage.delete(key);
                cleared++;
            }
        });
        
        if (cleared > 0 && typeof Logger !== 'undefined') {
            Logger.debug(`Cleared ${cleared} expired cache items`);
        }
        
        return cleared;
    },
    
    /**
     * Get cache statistics
     */
    getStats() {
        const now = Date.now();
        let total = 0;
        let expired = 0;
        let active = 0;
        
        this.storage.forEach((item) => {
            total++;
            if (now > item.expiry) {
                expired++;
            } else {
                active++;
            }
        });
        
        return {
            total,
            active,
            expired,
            size: this.storage.size
        };
    },
    
    /**
     * Get all cache keys
     */
    keys() {
        return Array.from(this.storage.keys());
    },
    
    /**
     * Set default TTL
     */
    setDefaultTTL(ttl) {
        this.defaultTTL = ttl;
    },
    
    /**
     * Auto-clean expired items periodically
     */
    startAutoClean(interval = 60000) { // Default: 1 minute
        if (this.autoCleanInterval) {
            clearInterval(this.autoCleanInterval);
        }
        
        this.autoCleanInterval = setInterval(() => {
            this.clearExpired();
        }, interval);
    },
    
    /**
     * Stop auto-clean
     */
    stopAutoClean() {
        if (this.autoCleanInterval) {
            clearInterval(this.autoCleanInterval);
            this.autoCleanInterval = null;
        }
    }
};

// Start auto-clean on initialization
if (typeof window !== 'undefined') {
    Cache.startAutoClean();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Cache;
}
