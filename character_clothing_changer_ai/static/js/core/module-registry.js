/**
 * Module Registry
 * ===============
 * Central registry for all application modules
 */

const ModuleRegistry = {
    /**
     * Registered modules
     */
    modules: new Map(),
    
    /**
     * Module metadata
     */
    metadata: new Map(),
    
    /**
     * Module dependencies
     */
    dependencies: new Map(),
    
    /**
     * Initialize registry
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Module registry initialized');
        }
        
        // Emit registry ready event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('module:registry_ready');
        }
    },
    
    /**
     * Register a module
     */
    register(name, module, metadata = {}) {
        if (!name || !module) {
            throw new Error('Module name and module are required');
        }
        
        this.modules.set(name, module);
        this.metadata.set(name, {
            name,
            registeredAt: new Date().toISOString(),
            ...metadata
        });
        
        // Store dependencies if provided
        if (metadata.dependencies) {
            this.dependencies.set(name, metadata.dependencies);
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Module registered: ${name}`);
        }
        
        // Emit module registered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('module:registered', { name, module, metadata });
        }
        
        return this;
    },
    
    /**
     * Get module
     */
    get(name) {
        return this.modules.get(name);
    },
    
    /**
     * Check if module is registered
     */
    has(name) {
        return this.modules.has(name);
    },
    
    /**
     * Get module metadata
     */
    getMetadata(name) {
        return this.metadata.get(name) || null;
    },
    
    /**
     * Get module dependencies
     */
    getDependencies(name) {
        return this.dependencies.get(name) || [];
    },
    
    /**
     * Get all registered modules
     */
    getAll() {
        return Array.from(this.modules.keys());
    },
    
    /**
     * Get all modules with metadata
     */
    getAllWithMetadata() {
        const result = [];
        this.modules.forEach((module, name) => {
            result.push({
                name,
                module,
                metadata: this.metadata.get(name),
                dependencies: this.dependencies.get(name) || []
            });
        });
        return result;
    },
    
    /**
     * Unregister module
     */
    unregister(name) {
        const removed = this.modules.delete(name);
        this.metadata.delete(name);
        this.dependencies.delete(name);
        
        if (removed && typeof Logger !== 'undefined') {
            Logger.debug(`Module unregistered: ${name}`);
        }
        
        // Emit module unregistered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('module:unregistered', { name });
        }
        
        return removed;
    },
    
    /**
     * Clear all modules
     */
    clear() {
        this.modules.clear();
        this.metadata.clear();
        this.dependencies.clear();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Module registry cleared');
        }
    },
    
    /**
     * Get registry stats
     */
    getStats() {
        return {
            totalModules: this.modules.size,
            modules: this.getAll(),
            categories: this.getCategories()
        };
    },
    
    /**
     * Get modules by category
     */
    getCategories() {
        const categories = {};
        this.metadata.forEach((meta, name) => {
            const category = meta.category || 'unknown';
            if (!categories[category]) {
                categories[category] = [];
            }
            categories[category].push(name);
        });
        return categories;
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    ModuleRegistry.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModuleRegistry;
}
