/**
 * Observable Module
 * ================
 * Reactive data structures with change detection
 */

const Observable = {
    /**
     * Create observable object
     */
    create(initialValue = {}) {
        const observers = new Set();
        let value = this.deepClone(initialValue);
        
        const observable = {
            /**
             * Get value
             */
            get() {
                return this.deepClone(value);
            },
            
            /**
             * Set value
             */
            set(newValue) {
                const oldValue = this.deepClone(value);
                value = this.deepClone(newValue);
                
                // Notify observers
                observers.forEach(observer => {
                    try {
                        observer(value, oldValue);
                    } catch (error) {
                        if (typeof Logger !== 'undefined') {
                            Logger.error('Error in observable observer', error);
                        }
                    }
                });
                
                return observable;
            },
            
            /**
             * Update value
             */
            update(updater) {
                const newValue = typeof updater === 'function' 
                    ? updater(value) 
                    : updater;
                return this.set(newValue);
            },
            
            /**
             * Subscribe to changes
             */
            subscribe(observer) {
                observers.add(observer);
                
                // Return unsubscribe function
                return () => {
                    observers.delete(observer);
                };
            },
            
            /**
             * Unsubscribe from changes
             */
            unsubscribe(observer) {
                observers.delete(observer);
            },
            
            /**
             * Get observers count
             */
            getObserverCount() {
                return observers.size;
            },
            
            /**
             * Clear all observers
             */
            clearObservers() {
                observers.clear();
            },
            
            /**
             * Deep clone helper
             */
            deepClone: this.deepClone
        };
        
        return observable;
    },
    
    /**
     * Create computed observable
     */
    computed(dependencies, computeFn) {
        const computed = this.create();
        const unsubscribes = [];
        
        // Subscribe to all dependencies
        dependencies.forEach(dep => {
            if (dep && typeof dep.subscribe === 'function') {
                const unsubscribe = dep.subscribe(() => {
                    computed.set(computeFn());
                });
                unsubscribes.push(unsubscribe);
            }
        });
        
        // Initial computation
        computed.set(computeFn());
        
        // Add cleanup method
        computed.cleanup = () => {
            unsubscribes.forEach(unsub => unsub());
        };
        
        return computed;
    },
    
    /**
     * Deep clone helper
     */
    deepClone(obj) {
        if (obj === null || typeof obj !== 'object') {
            return obj;
        }
        
        if (obj instanceof Date) {
            return new Date(obj.getTime());
        }
        
        if (Array.isArray(obj)) {
            return obj.map(item => this.deepClone(item));
        }
        
        const cloned = {};
        Object.keys(obj).forEach(key => {
            cloned[key] = this.deepClone(obj[key]);
        });
        
        return cloned;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Observable;
}

