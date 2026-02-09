/**
 * Middleware Chain Module
 * =======================
 * Chain of responsibility pattern for request/response processing
 */

const MiddlewareChain = {
    /**
     * Create middleware chain
     */
    create() {
        const middlewares = [];
        
        const chain = {
            /**
             * Add middleware
             */
            use(middleware) {
                if (typeof middleware !== 'function') {
                    throw new Error('Middleware must be a function');
                }
                
                middlewares.push(middleware);
                return chain;
            },
            
            /**
             * Execute chain
             */
            async execute(context, initialValue) {
                let value = initialValue;
                let index = 0;
                
                const next = async () => {
                    if (index >= middlewares.length) {
                        return value;
                    }
                    
                    const middleware = middlewares[index++];
                    
                    try {
                        const result = await middleware(context, value, next);
                        if (result !== undefined) {
                            value = result;
                        }
                        return value;
                    } catch (error) {
                        if (typeof Logger !== 'undefined') {
                            Logger.error('Middleware error', error);
                        }
                        throw error;
                    }
                };
                
                return await next();
            },
            
            /**
             * Get middleware count
             */
            getCount() {
                return middlewares.length;
            },
            
            /**
             * Clear all middlewares
             */
            clear() {
                middlewares.length = 0;
            }
        };
        
        return chain;
    },
    
    /**
     * Create request middleware chain
     */
    createRequestChain() {
        return this.create();
    },
    
    /**
     * Create response middleware chain
     */
    createResponseChain() {
        return this.create();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MiddlewareChain;
}

