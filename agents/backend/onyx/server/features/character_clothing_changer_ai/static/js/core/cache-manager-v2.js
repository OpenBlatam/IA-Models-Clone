/**
 * Cache Manager V2 Module
 * ======================
 * Advanced caching system with strategies and TTL
 */

const CacheManagerV2 = {
    /**
     * Cache storage
     */
    cache: new Map(),
    
    /**
     * Cache strategies
     */
    strategies: {
        LRU: 'lru',
        LFU: 'lfu',
        FIFO: 'fifo',
        TTL: 'ttl'
    },
    
    /**
     * Current strategy
     */
    strategy: 'lru',
    
    /**
     * Max cache size
     */
    maxSize: 100,
    
    /**
     * Default TTL (ms)
     */
    defaultTTL: 3600000, // 1 hour
    
    /**
     * Access order (for LRU)
     */
    accessOrder: [],
    
    /**
     * Access frequency (for LFU)
     */
    accessFrequency: new Map(),
    
    /**
     * Initialize cache manager
     */
    init(options = {}) {
        this.strategy = options.strategy || this.strategy;
        this.maxSize = options.maxSize || this.maxSize;
        this.defaultTTL = options.defaultTTL || this.defaultTTL;
        
        // Load from storage
        this.loadFromStorage();
        
        // Setup cleanup interval
        this.setupCleanup();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Cache Manager V2 initialized', { strategy: this.strategy, maxSize: this.maxSize });
        }
    },
    
    /**
     * Set cache item
     */
    set(key, value, options = {}) {
        const ttl = options.ttl || this.defaultTTL;
        const expiresAt = Date.now() + ttl;
        
        const item = {
            value,
            expiresAt,
            createdAt: Date.now(),
            accessCount: 0,
            lastAccessed: Date.now()
        };
        
        // Check if cache is full
        if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
            this.evict();
        }
        
        this.cache.set(key, item);
        
        // Update access order
        this.updateAccessOrder(key);
        
        // Save to storage
        this.saveToStorage();
        
        // Emit cache set event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('cache:set', { key, value });
        }
        
        return this;
    },
    
    /**
     * Get cache item
     */
    get(key) {
        const item = this.cache.get(key);
        
        if (!item) {
            return null;
        }
        
        // Check if expired
        if (Date.now() > item.expiresAt) {
            this.delete(key);
            return null;
        }
        
        // Update access info
        item.accessCount++;
        item.lastAccessed = Date.now();
        this.updateAccessOrder(key);
        this.updateAccessFrequency(key);
        
        // Save to storage
        this.saveToStorage();
        
        return item.value;
    },
    
    /**
     * Check if key exists
     */
    has(key) {
        const item = this.cache.get(key);
        if (!item) {
            return false;
        }
        
        // Check if expired
        if (Date.now() > item.expiresAt) {
            this.delete(key);
            return false;
        }
        
        return true;
    },
    
    /**
     * Delete cache item
     */
    delete(key) {
        const deleted = this.cache.delete(key);
        this.accessOrder = this.accessOrder.filter(k => k !== key);
        this.accessFrequency.delete(key);
        
        // Save to storage
        this.saveToStorage();
        
        // Emit cache delete event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('cache:delete', { key });
        }
        
        return deleted;
    },
    
    /**
     * Clear cache
     */
    clear() {
        this.cache.clear();
        this.accessOrder = [];
        this.accessFrequency.clear();
        
        // Clear storage
        try {
            localStorage.removeItem('cache_v2');
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to clear cache from storage', error);
            }
        }
        
        // Emit cache clear event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('cache:clear');
        }
    },
    
    /**
     * Evict item based on strategy
     */
    evict() {
        if (this.cache.size === 0) {
            return;
        }
        
        let keyToEvict;
        
        switch (this.strategy) {
            case 'lru':
                // Least recently used
                keyToEvict = this.accessOrder[0];
                break;
            case 'lfu':
                // Least frequently used
                let minFreq = Infinity;
                this.accessFrequency.forEach((freq, key) => {
                    if (freq < minFreq) {
                        minFreq = freq;
                        keyToEvict = key;
                    }
                });
                break;
            case 'fifo':
                // First in first out
                const oldest = Array.from(this.cache.entries())
                    .sort((a, b) => a[1].createdAt - b[1].createdAt)[0];
                keyToEvict = oldest[0];
                break;
            case 'ttl':
                // Expired items first, then oldest
                const expired = Array.from(this.cache.entries())
                    .find(([_, item]) => Date.now() > item.expiresAt);
                if (expired) {
                    keyToEvict = expired[0];
                } else {
                    const oldest = Array.from(this.cache.entries())
                        .sort((a, b) => a[1].expiresAt - b[1].expiresAt)[0];
                    keyToEvict = oldest[0];
                }
                break;
            default:
                keyToEvict = this.accessOrder[0];
        }
        
        if (keyToEvict) {
            this.delete(keyToEvict);
        }
    },
    
    /**
     * Update access order
     */
    updateAccessOrder(key) {
        // Remove from current position
        this.accessOrder = this.accessOrder.filter(k => k !== key);
        // Add to end (most recently used)
        this.accessOrder.push(key);
    },
    
    /**
     * Update access frequency
     */
    updateAccessFrequency(key) {
        const current = this.accessFrequency.get(key) || 0;
        this.accessFrequency.set(key, current + 1);
    },
    
    /**
     * Setup cleanup interval
     */
    setupCleanup() {
        setInterval(() => {
            this.cleanup();
        }, 60000); // Every minute
    },
    
    /**
     * Cleanup expired items
     */
    cleanup() {
        const now = Date.now();
        const expired = [];
        
        this.cache.forEach((item, key) => {
            if (now > item.expiresAt) {
                expired.push(key);
            }
        });
        
        expired.forEach(key => this.delete(key));
        
        if (expired.length > 0 && typeof Logger !== 'undefined') {
            Logger.debug(`Cleaned up ${expired.length} expired cache items`);
        }
    },
    
    /**
     * Save to storage
     */
    saveToStorage() {
        try {
            const data = {
                items: Array.from(this.cache.entries()).map(([key, item]) => [
                    key,
                    {
                        value: item.value,
                        expiresAt: item.expiresAt,
                        createdAt: item.createdAt,
                        accessCount: item.accessCount,
                        lastAccessed: item.lastAccessed
                    }
                ]),
                accessOrder: this.accessOrder,
                accessFrequency: Array.from(this.accessFrequency.entries())
            };
            
            localStorage.setItem('cache_v2', JSON.stringify(data));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save cache to storage', error);
            }
        }
    },
    
    /**
     * Load from storage
     */
    loadFromStorage() {
        try {
            const stored = localStorage.getItem('cache_v2');
            if (!stored) {
                return;
            }
            
            const data = JSON.parse(stored);
            const now = Date.now();
            
            // Load items (filter expired)
            data.items.forEach(([key, item]) => {
                if (now < item.expiresAt) {
                    this.cache.set(key, item);
                }
            });
            
            // Load access order
            this.accessOrder = data.accessOrder || [];
            
            // Load access frequency
            if (data.accessFrequency) {
                this.accessFrequency = new Map(data.accessFrequency);
            }
            
            if (typeof Logger !== 'undefined') {
                Logger.debug(`Loaded ${this.cache.size} items from cache storage`);
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to load cache from storage', error);
            }
        }
    },
    
    /**
     * Get cache stats
     */
    getStats() {
        const now = Date.now();
        let expired = 0;
        let totalAccess = 0;
        
        this.cache.forEach(item => {
            if (now > item.expiresAt) {
                expired++;
            }
            totalAccess += item.accessCount;
        });
        
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            strategy: this.strategy,
            expired,
            totalAccess,
            hitRate: this.calculateHitRate()
        };
    },
    
    /**
     * Calculate hit rate
     */
    calculateHitRate() {
        // This would need to track hits/misses over time
        // For now, return a simple calculation
        return 0; // Placeholder
    },
    
    /**
     * Get all keys
     */
    keys() {
        return Array.from(this.cache.keys());
    },
    
    /**
     * Get all values
     */
    values() {
        return Array.from(this.cache.values()).map(item => item.value);
    },
    
    /**
     * Get all entries
     */
    entries() {
        return Array.from(this.cache.entries()).map(([key, item]) => [key, item.value]);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CacheManagerV2;
}

