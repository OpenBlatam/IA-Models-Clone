/**
 * Module Loader
 * =============
 * Advanced module loading system with dependency management
 */

const ModuleLoader = {
    /**
     * Loaded modules registry
     */
    loadedModules: new Set(),
    
    /**
     * Module dependencies
     */
    dependencies: {
        'StateManager': ['EventBus', 'Logger'],
        'ErrorHandler': ['Logger'],
        'API': ['CONFIG', 'Cache', 'EventBus', 'PerformanceMonitor'],
        'Form': ['FormDataBuilder', 'Validator', 'EventBus', 'StateManager'],
        'GalleryManager': ['Storage', 'ItemRenderer', 'Favorites', 'EventBus'],
        'HistoryManager': ['Storage', 'ItemRenderer', 'Favorites', 'EventBus'],
        'Filters': ['SearchFilter', 'ItemRenderer', 'Debounce'],
        'Analytics': ['EventBus', 'Logger'],
        'PerformanceMonitor': ['EventBus', 'Logger']
    },
    
    /**
     * Load a module
     */
    async loadModule(moduleName, path = null) {
        // Check if already loaded
        if (this.loadedModules.has(moduleName)) {
            return true;
        }
        
        // Check dependencies
        const deps = this.dependencies[moduleName] || [];
        for (const dep of deps) {
            if (!this.isModuleLoaded(dep)) {
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Module ${moduleName} requires ${dep}, but it's not loaded`);
                }
            }
        }
        
        // Load module script
        if (path) {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = path;
                script.onload = () => {
                    this.loadedModules.add(moduleName);
                    if (typeof Logger !== 'undefined') {
                        Logger.debug(`Module loaded: ${moduleName}`);
                    }
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('module:loaded', moduleName);
                    }
                    resolve(true);
                };
                script.onerror = () => {
                    if (typeof Logger !== 'undefined') {
                        Logger.error(`Failed to load module: ${moduleName}`);
                    }
                    reject(new Error(`Failed to load module: ${moduleName}`));
                };
                document.head.appendChild(script);
            });
        }
        
        return false;
    },
    
    /**
     * Check if module is loaded
     */
    isModuleLoaded(moduleName) {
        // Check registry
        if (this.loadedModules.has(moduleName)) {
            return true;
        }
        
        // Check global scope
        return typeof window[moduleName] !== 'undefined';
    },
    
    /**
     * Load modules in order respecting dependencies
     */
    async loadModules(modules) {
        const loaded = new Set();
        const loading = new Set();
        
        const loadWithDeps = async (moduleName, path) => {
            if (loaded.has(moduleName) || loading.has(moduleName)) {
                return;
            }
            
            loading.add(moduleName);
            
            // Load dependencies first
            const deps = this.dependencies[moduleName] || [];
            for (const dep of deps) {
                if (!loaded.has(dep) && !loading.has(dep)) {
                    // Try to find dependency path
                    const depPath = this.findModulePath(dep);
                    if (depPath) {
                        await loadWithDeps(dep, depPath);
                    }
                }
            }
            
            // Load module
            await this.loadModule(moduleName, path);
            loaded.add(moduleName);
            loading.delete(moduleName);
        };
        
        for (const [moduleName, path] of Object.entries(modules)) {
            await loadWithDeps(moduleName, path);
        }
    },
    
    /**
     * Find module path based on naming conventions
     */
    findModulePath(moduleName) {
        // Convert module name to file path
        const fileName = moduleName
            .replace(/([A-Z])/g, '-$1')
            .toLowerCase()
            .substring(1) + '.js';
        
        // Try different locations
        const locations = [
            `static/js/${fileName}`,
            `static/js/core/${fileName}`,
            `static/js/utils/${fileName}`,
            `static/js/ui/${fileName}`,
            `static/js/features/${fileName}`,
            `static/js/renderers/${fileName}`
        ];
        
        return locations[0]; // Return first, actual check would need to verify file exists
    },
    
    /**
     * Get loaded modules list
     */
    getLoadedModules() {
        return Array.from(this.loadedModules);
    },
    
    /**
     * Initialize module loader
     */
    init() {
        // Mark already loaded modules
        const globalModules = [
            'CONFIG', 'Storage', 'Logger', 'EventBus', 'StateManager',
            'ErrorHandler', 'Cache', 'API', 'Validator', 'Debounce',
            'SearchFilter', 'ItemRenderer', 'Analytics', 'PerformanceMonitor',
            'Notifications', 'Favorites', 'Filters', 'Shortcuts', 'Stats',
            'UI', 'Form', 'GalleryManager', 'HistoryManager', 'Comparison',
            'ProgressBar', 'ImageAnalyzer'
        ];
        
        globalModules.forEach(module => {
            if (typeof window[module] !== 'undefined') {
                this.loadedModules.add(module);
            }
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Module loader initialized. Loaded modules: ${this.loadedModules.size}`);
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    ModuleLoader.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModuleLoader;
}
