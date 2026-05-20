/**
 * Middleware System Module
 * ========================
 * Request/response middleware pipeline
 */

const Middleware = {
    /**
     * Middleware stack
     */
    stack: [],
    
    /**
     * Add middleware to stack
     */
    use(middleware) {
        if (typeof middleware !== 'function') {
            throw new Error('Middleware must be a function');
        }
        
        this.stack.push(middleware);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug('Middleware added to stack');
        }
        
        return this;
    },
    
    /**
     * Execute middleware stack
     */
    async execute(context, next) {
        let index = 0;
        
        const run = async (i) => {
            if (i >= this.stack.length) {
                return next ? await next() : undefined;
            }
            
            const middleware = this.stack[i];
            
            return await middleware(context, async () => {
                return await run(i + 1);
            });
        };
        
        return await run(0);
    },
    
    /**
     * Clear middleware stack
     */
    clear() {
        this.stack = [];
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Middleware stack cleared');
        }
    },
    
    /**
     * Get middleware count
     */
    count() {
        return this.stack.length;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Middleware;
}

