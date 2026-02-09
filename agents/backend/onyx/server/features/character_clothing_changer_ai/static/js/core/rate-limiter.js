/**
 * Rate Limiter Module
 * ==================
 * Rate limiting for API calls and operations
 */

const RateLimiter = {
    /**
     * Rate limiters
     */
    limiters: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        maxRequests: 10,
        windowMs: 60000, // 1 minute
        strategy: 'sliding' // 'sliding' or 'fixed'
    },
    
    /**
     * Create rate limiter
     */
    create(name, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const limiter = {
            name,
            maxRequests: config.maxRequests,
            windowMs: config.windowMs,
            strategy: config.strategy,
            requests: [],
            blocked: false,
            blockUntil: null
        };
        
        this.limiters.set(name, limiter);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Rate limiter created: ${name}`, config);
        }
        
        return limiter;
    },
    
    /**
     * Check if request is allowed
     */
    isAllowed(name) {
        const limiter = this.limiters.get(name);
        if (!limiter) {
            return true; // No limiter, allow
        }
        
        const now = Date.now();
        
        // Check if blocked
        if (limiter.blocked && limiter.blockUntil && now < limiter.blockUntil) {
            return false;
        }
        
        // Reset block if expired
        if (limiter.blocked && limiter.blockUntil && now >= limiter.blockUntil) {
            limiter.blocked = false;
            limiter.blockUntil = null;
        }
        
        // Clean old requests based on strategy
        this.cleanRequests(limiter, now);
        
        // Check if limit exceeded
        if (limiter.requests.length >= limiter.maxRequests) {
            // Block for window duration
            limiter.blocked = true;
            limiter.blockUntil = now + limiter.windowMs;
            
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Rate limit exceeded: ${name}`);
            }
            
            // Emit rate limit exceeded event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('rate-limit:exceeded', { name, limiter });
            }
            
            return false;
        }
        
        // Add request
        limiter.requests.push(now);
        
        return true;
    },
    
    /**
     * Clean old requests
     */
    cleanRequests(limiter, now) {
        if (limiter.strategy === 'sliding') {
            // Remove requests outside window
            limiter.requests = limiter.requests.filter(
                timestamp => now - timestamp < limiter.windowMs
            );
        } else {
            // Fixed window: remove all if window expired
            if (limiter.requests.length > 0) {
                const oldest = limiter.requests[0];
                if (now - oldest >= limiter.windowMs) {
                    limiter.requests = [];
                }
            }
        }
    },
    
    /**
     * Get remaining requests
     */
    getRemaining(name) {
        const limiter = this.limiters.get(name);
        if (!limiter) {
            return Infinity;
        }
        
        const now = Date.now();
        this.cleanRequests(limiter, now);
        
        return Math.max(0, limiter.maxRequests - limiter.requests.length);
    },
    
    /**
     * Get reset time
     */
    getResetTime(name) {
        const limiter = this.limiters.get(name);
        if (!limiter) {
            return null;
        }
        
        if (limiter.requests.length === 0) {
            return null;
        }
        
        const oldest = limiter.requests[0];
        return oldest + limiter.windowMs;
    },
    
    /**
     * Reset limiter
     */
    reset(name) {
        const limiter = this.limiters.get(name);
        if (!limiter) {
            return false;
        }
        
        limiter.requests = [];
        limiter.blocked = false;
        limiter.blockUntil = null;
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Rate limiter reset: ${name}`);
        }
        
        return true;
    },
    
    /**
     * Get limiter status
     */
    getStatus(name) {
        const limiter = this.limiters.get(name);
        if (!limiter) {
            return null;
        }
        
        const now = Date.now();
        this.cleanRequests(limiter, now);
        
        return {
            name: limiter.name,
            maxRequests: limiter.maxRequests,
            currentRequests: limiter.requests.length,
            remaining: limiter.maxRequests - limiter.requests.length,
            blocked: limiter.blocked,
            blockUntil: limiter.blockUntil,
            resetTime: this.getResetTime(name)
        };
    },
    
    /**
     * Get all limiters
     */
    getAll() {
        return Array.from(this.limiters.values());
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RateLimiter;
}

