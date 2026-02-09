/**
 * Module Loader V2
 * ================
 * Advanced module loading system with dependency resolution
 */

const ModuleLoaderV2 = {
    /**
     * Loaded modules
     */
    loadedModules: new Set(),
    
    /**
     * Loading promises
     */
    loadingPromises: new Map(),
    
    /**
     * Module paths
     */
    paths: {
        core: 'static/js/core',
        utils: 'static/js/utils',
        ui: 'static/js/ui',
        features: 'static/js/features',
        renderers: 'static/js/renderers',
        system: 'static/js'
    },
    
    /**
     * Initialize loader
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Module Loader V2 initialized');
        }
        
        // Emit loader ready event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('module:loader_ready');
        }
    },
    
    /**
     * Load a module
     */
    async loadModule(moduleName, category = 'core') {
        // Check if already loaded
        if (this.loadedModules.has(moduleName)) {
            return this.getModule(moduleName);
        }
        
        // Check if already loading
        if (this.loadingPromises.has(moduleName)) {
            return this.loadingPromises.get(moduleName);
        }
        
        // Create loading promise
        const loadPromise = this._loadModuleFile(moduleName, category);
        this.loadingPromises.set(moduleName, loadPromise);
        
        try {
            const module = await loadPromise;
            this.loadedModules.add(moduleName);
            this.loadingPromises.delete(moduleName);
            
            // Register in module registry if available
            if (typeof ModuleRegistry !== 'undefined') {
                ModuleRegistry.register(moduleName, module, {
                    category,
                    loadedAt: new Date().toISOString()
                });
            }
            
            if (typeof Logger !== 'undefined') {
                Logger.debug(`Module loaded: ${category}/${moduleName}`);
            }
            
            // Emit module loaded event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('module:loaded', { name: moduleName, category });
            }
            
            return module;
        } catch (error) {
            this.loadingPromises.delete(moduleName);
            
            if (typeof Logger !== 'undefined') {
                Logger.error(`Failed to load module ${category}/${moduleName}`, error);
            }
            
            throw error;
        }
    },
    
    /**
     * Load module file
     */
    _loadModuleFile(moduleName, category) {
        return new Promise((resolve, reject) => {
            const path = this.paths[category] || this.paths.core;
            const script = document.createElement('script');
            script.src = `${path}/${moduleName}.js`;
            script.async = true;
            
            script.onload = () => {
                // Try to get module from global scope
                const module = this._getModuleFromGlobal(moduleName);
                if (module) {
                    resolve(module);
                } else {
                    // Module loaded but not found in global scope
                    // This is OK for modules that don't export to global
                    resolve({});
                }
            };
            
            script.onerror = () => {
                reject(new Error(`Failed to load module: ${path}/${moduleName}.js`));
            };
            
            document.head.appendChild(script);
        });
    },
    
    /**
     * Get module from global scope
     */
    _getModuleFromGlobal(moduleName) {
        // Try different naming conventions
        const names = [
            moduleName,
            this._toPascalCase(moduleName),
            this._toCamelCase(moduleName)
        ];
        
        for (const name of names) {
            if (typeof window[name] !== 'undefined') {
                return window[name];
            }
        }
        
        return null;
    },
    
    /**
     * Convert to PascalCase
     */
    _toPascalCase(str) {
        return str.replace(/(?:^\w|[A-Z]|\b\w)/g, (word) => {
            return word.toUpperCase();
        }).replace(/\s+/g, '');
    },
    
    /**
     * Convert to camelCase
     */
    _toCamelCase(str) {
        return str.replace(/-([a-z])/g, (g) => g[1].toUpperCase());
    },
    
    /**
     * Load multiple modules
     */
    async loadModules(modules, category = 'core') {
        const promises = modules.map(moduleName => 
            this.loadModule(moduleName, category).catch(error => {
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Module ${moduleName} failed to load:`, error);
                }
                return null;
            })
        );
        
        const results = await Promise.all(promises);
        return results.filter(r => r !== null);
    },
    
    /**
     * Load modules in order
     */
    async loadModulesSequential(modules, category = 'core') {
        const results = [];
        for (const moduleName of modules) {
            try {
                const module = await this.loadModule(moduleName, category);
                results.push(module);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Module ${moduleName} failed to load:`, error);
                }
            }
        }
        return results;
    },
    
    /**
     * Check if module is loaded
     */
    isLoaded(moduleName) {
        return this.loadedModules.has(moduleName);
    },
    
    /**
     * Get loaded module
     */
    getModule(moduleName) {
        if (typeof ModuleRegistry !== 'undefined') {
            return ModuleRegistry.get(moduleName);
        }
        return this._getModuleFromGlobal(moduleName);
    },
    
    /**
     * Get all loaded modules
     */
    getLoadedModules() {
        return Array.from(this.loadedModules);
    },
    
    /**
     * Set module path
     */
    setPath(category, path) {
        this.paths[category] = path;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModuleLoaderV2;
}
