/**
 * Plugin System Module
 * ====================
 * Extensible plugin architecture
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
        if (!plugin.name || !plugin.version) {
            throw new Error('Plugin must have name and version');
        }
        
        // Validate plugin
        if (plugin.init && typeof plugin.init !== 'function') {
            throw new Error('Plugin init must be a function');
        }
        
        // Register plugin
        this.plugins.set(name, {
            ...plugin,
            registeredAt: Date.now(),
            enabled: true
        });
        
        // Initialize plugin
        if (plugin.init) {
            try {
                plugin.init();
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error initializing plugin ${name}:`, error);
                }
                throw error;
            }
        }
        
        // Emit plugin registered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:registered', { name, plugin });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Plugin registered: ${name} v${plugin.version}`);
        }
        
        return true;
    },
    
    /**
     * Unregister a plugin
     */
    unregister(name) {
        const plugin = this.plugins.get(name);
        if (!plugin) {
            return false;
        }
        
        // Call cleanup if available
        if (plugin.cleanup && typeof plugin.cleanup === 'function') {
            try {
                plugin.cleanup();
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error cleaning up plugin ${name}:`, error);
                }
            }
        }
        
        this.plugins.delete(name);
        
        // Emit plugin unregistered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:unregistered', { name });
        }
        
        return true;
    },
    
    /**
     * Enable a plugin
     */
    enable(name) {
        const plugin = this.plugins.get(name);
        if (plugin) {
            plugin.enabled = true;
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('plugin:enabled', { name });
            }
            return true;
        }
        return false;
    },
    
    /**
     * Disable a plugin
     */
    disable(name) {
        const plugin = this.plugins.get(name);
        if (plugin) {
            plugin.enabled = false;
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('plugin:disabled', { name });
            }
            return true;
        }
        return false;
    },
    
    /**
     * Register a hook
     */
    registerHook(hookName, callback, priority = 10) {
        if (!this.hooks.has(hookName)) {
            this.hooks.set(hookName, []);
        }
        
        const hooks = this.hooks.get(hookName);
        hooks.push({ callback, priority });
        
        // Sort by priority
        hooks.sort((a, b) => a.priority - b.priority);
        
        return () => {
            const index = hooks.findIndex(h => h.callback === callback);
            if (index > -1) {
                hooks.splice(index, 1);
            }
        };
    },
    
    /**
     * Execute a hook
     */
    executeHook(hookName, ...args) {
        const hooks = this.hooks.get(hookName);
        if (!hooks || hooks.length === 0) {
            return args[0]; // Return first argument if no hooks
        }
        
        let result = args[0];
        
        for (const { callback } of hooks) {
            try {
                const hookResult = callback(result, ...args.slice(1));
                if (hookResult !== undefined) {
                    result = hookResult;
                }
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error in hook ${hookName}:`, error);
                }
                if (typeof ErrorHandler !== 'undefined') {
                    ErrorHandler.handle(error, { context: `hook:${hookName}` });
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
     * Get enabled plugins
     */
    getEnabledPlugins() {
        return Array.from(this.plugins.values()).filter(p => p.enabled);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PluginSystem;
}

