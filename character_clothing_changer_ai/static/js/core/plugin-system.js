/**
 * Plugin System
 * =============
 * Plugin system for extending application functionality
 */

const PluginSystem = {
    /**
     * Registered plugins
     */
    plugins: new Map(),
    
    /**
     * Plugin hooks
     */
    hooks: new Map(),
    
    /**
     * Register a plugin
     */
    register(name, plugin) {
        if (this.plugins.has(name)) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Plugin ${name} is already registered`);
            }
            return false;
        }
        
        // Validate plugin
        if (!plugin.name || !plugin.version) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Plugin ${name} is missing required fields`);
            }
            return false;
        }
        
        // Initialize plugin
        if (plugin.init) {
            try {
                plugin.init();
            } catch (e) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Failed to initialize plugin ${name}:`, e);
                }
                return false;
            }
        }
        
        this.plugins.set(name, plugin);
        
        // Register hooks
        if (plugin.hooks) {
            Object.keys(plugin.hooks).forEach(hookName => {
                this.registerHook(hookName, plugin.hooks[hookName]);
            });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Plugin registered: ${name} v${plugin.version}`);
        }
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:registered', name, plugin);
        }
        
        return true;
    },
    
    /**
     * Unregister a plugin
     */
    unregister(name) {
        if (!this.plugins.has(name)) {
            return false;
        }
        
        const plugin = this.plugins.get(name);
        
        // Cleanup plugin
        if (plugin.cleanup) {
            try {
                plugin.cleanup();
            } catch (e) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Failed to cleanup plugin ${name}:`, e);
                }
            }
        }
        
        // Remove hooks
        if (plugin.hooks) {
            Object.keys(plugin.hooks).forEach(hookName => {
                this.unregisterHook(hookName, plugin.hooks[hookName]);
            });
        }
        
        this.plugins.delete(name);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Plugin unregistered: ${name}`);
        }
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:unregistered', name);
        }
        
        return true;
    },
    
    /**
     * Register a hook
     */
    registerHook(hookName, callback) {
        if (!this.hooks.has(hookName)) {
            this.hooks.set(hookName, []);
        }
        
        this.hooks.get(hookName).push(callback);
    },
    
    /**
     * Unregister a hook
     */
    unregisterHook(hookName, callback) {
        if (!this.hooks.has(hookName)) {
            return;
        }
        
        const callbacks = this.hooks.get(hookName);
        const index = callbacks.indexOf(callback);
        if (index > -1) {
            callbacks.splice(index, 1);
        }
    },
    
    /**
     * Execute hooks
     */
    executeHook(hookName, ...args) {
        if (!this.hooks.has(hookName)) {
            return args[0]; // Return first argument if no hooks
        }
        
        let result = args[0];
        const callbacks = this.hooks.get(hookName);
        
        for (const callback of callbacks) {
            try {
                const callbackResult = callback(result, ...args.slice(1));
                if (callbackResult !== undefined) {
                    result = callbackResult;
                }
            } catch (e) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Hook error in ${hookName}:`, e);
                }
            }
        }
        
        return result;
    },
    
    /**
     * Get plugin
     */
    getPlugin(name) {
        return this.plugins.get(name);
    },
    
    /**
     * Get all plugins
     */
    getAllPlugins() {
        return Array.from(this.plugins.values());
    },
    
    /**
     * Check if plugin is registered
     */
    hasPlugin(name) {
        return this.plugins.has(name);
    },
    
    /**
     * Get plugin count
     */
    getPluginCount() {
        return this.plugins.size;
    }
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PluginSystem;
}

