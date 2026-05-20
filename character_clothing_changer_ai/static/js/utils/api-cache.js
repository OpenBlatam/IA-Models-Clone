/**
 * API Cache Module
 * ================
 * Handles caching for API requests
 */

const APICache = {
    /**
     * Get cache key for request
     */
    getCacheKey(endpoint, options = {}) {
        return `api:${endpoint}:${JSON.stringify(options)}`;
    },

    /**
     * Check if request should be cached
     */
    shouldCache(method, useCache) {
        if (useCache === false) return false;
        return method !== 'POST' && method !== 'PUT' && method !== 'DELETE';
    },

    /**
     * Get cached response
     */
    get(cacheKey) {
        if (typeof Cache === 'undefined') return null;
        
        const cached = Cache.get(cacheKey);
        if (cached) {
            if (typeof Logger !== 'undefined') {
                Logger.debug(`API cache hit: ${cacheKey}`);
            }
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('api:cached', { cacheKey, data: cached });
            }
            return cached;
        }
        return null;
    },

    /**
     * Set cached response
     */
    set(cacheKey, response, ttl = 5 * 60 * 1000) {
        if (typeof Cache === 'undefined') return;
        
        Cache.set(cacheKey, response, ttl);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APICache;
}

