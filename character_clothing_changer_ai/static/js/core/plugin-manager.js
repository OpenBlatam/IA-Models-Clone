/**
 * Plugin Manager Module
 * =====================
 * Manages plugins and extensions for the application
 */

const PluginManager = {
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
        if (!plugin || typeof plugin !== 'object') {
            throw new Error(`Invalid plugin: ${name}`);
        }
        
        if (!plugin.name) {
            plugin.name = name;
        }
        
        if (!plugin.version) {
            plugin.version = '1.0.0';
        }
        
        // Validate plugin
        this.validatePlugin(plugin);
        
        // Initialize plugin if init method exists
        if (typeof plugin.init === 'function') {
            try {
                plugin.init();
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error initializing plugin ${name}:`, error);
                }
                throw error;
            }
        }
        
        this.plugins.set(name, plugin);
        
        // Register plugin hooks
        if (plugin.hooks) {
            Object.keys(plugin.hooks).forEach(hookName => {
                this.registerHook(hookName, plugin.hooks[hookName], name);
            });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Plugin registered: ${name} v${plugin.version}`);
        }
        
        // Emit plugin registered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:registered', { name, plugin });
        }
        
        return this;
    },
    
    /**
     * Unregister a plugin
     */
    unregister(name) {
        const plugin = this.plugins.get(name);
        if (!plugin) {
            return false;
        }
        
        // Cleanup plugin if cleanup method exists
        if (typeof plugin.cleanup === 'function') {
            try {
                plugin.cleanup();
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error cleaning up plugin ${name}:`, error);
                }
            }
        }
        
        // Remove plugin hooks
        this.hooks.forEach((handlers, hookName) => {
            this.hooks.set(
                hookName,
                handlers.filter(h => h.plugin !== name)
            );
        });
        
        this.plugins.delete(name);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Plugin unregistered: ${name}`);
        }
        
        // Emit plugin unregistered event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:unregistered', { name });
        }
        
        return true;
    },
    
    /**
     * Get a plugin
     */
    get(name) {
        return this.plugins.get(name);
    },
    
    /**
     * Check if plugin is registered
     */
    has(name) {
        return this.plugins.has(name);
    },
    
    /**
     * Get all plugins
     */
    getAll() {
        return Array.from(this.plugins.values());
    },
    
    /**
     * Register a hook
     */
    registerHook(hookName, handler, pluginName) {
        if (!this.hooks.has(hookName)) {
            this.hooks.set(hookName, []);
        }
        
        this.hooks.get(hookName).push({
            handler,
            plugin: pluginName
        });
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Hook registered: ${hookName} by ${pluginName}`);
        }
    },
    
    /**
     * Execute a hook
     */
    executeHook(hookName, ...args) {
        if (!this.hooks.has(hookName)) {
            return args[0]; // Return first argument if no hooks
        }
        
        const handlers = this.hooks.get(hookName);
        let result = args[0];
        
        handlers.forEach(({ handler, plugin }) => {
            try {
                const hookResult = handler(result, ...args.slice(1));
                if (hookResult !== undefined) {
                    result = hookResult;
                }
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error executing hook ${hookName} in plugin ${plugin}:`, error);
                }
            }
        });
        
        return result;
    },
    
    /**
     * Execute hook asynchronously
     */
    async executeHookAsync(hookName, ...args) {
        if (!this.hooks.has(hookName)) {
            return args[0];
        }
        
        const handlers = this.hooks.get(hookName);
        let result = args[0];
        
        for (const { handler, plugin } of handlers) {
            try {
                const hookResult = await handler(result, ...args.slice(1));
                if (hookResult !== undefined) {
                    result = hookResult;
                }
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error executing async hook ${hookName} in plugin ${plugin}:`, error);
                }
            }
        }
        
        return result;
    },
    
    /**
     * Validate plugin structure
     */
    validatePlugin(plugin) {
        if (!plugin.name) {
            throw new Error('Plugin must have a name');
        }
        
        if (!plugin.version) {
            throw new Error('Plugin must have a version');
        }
        
        // Optional: validate plugin API version compatibility
        if (plugin.apiVersion) {
            const currentAPIVersion = '1.0.0'; // Current API version
            // Add version compatibility check if needed
        }
        
        return true;
    },
    
    /**
     * Get plugin information
     */
    getPluginInfo(name) {
        const plugin = this.plugins.get(name);
        if (!plugin) {
            return null;
        }
        
        return {
            name: plugin.name,
            version: plugin.version,
            description: plugin.description,
            author: plugin.author,
            enabled: plugin.enabled !== false,
            hooks: Array.from(this.hooks.entries())
                .filter(([_, handlers]) => handlers.some(h => h.plugin === name))
                .map(([hookName]) => hookName)
        };
    },
    
    /**
     * Enable/disable plugin
     */
    setPluginEnabled(name, enabled) {
        const plugin = this.plugins.get(name);
        if (!plugin) {
            return false;
        }
        
        plugin.enabled = enabled;
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Plugin ${name} ${enabled ? 'enabled' : 'disabled'}`);
        }
        
        // Emit plugin state change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('plugin:state_changed', { name, enabled });
        }
        
        return true;
    },
    
    /**
     * Get all hooks
     */
    getHooks() {
        return Array.from(this.hooks.keys());
    },
    
    /**
     * Clear all plugins
     */
    clear() {
        const pluginNames = Array.from(this.plugins.keys());
        pluginNames.forEach(name => this.unregister(name));
        this.hooks.clear();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('All plugins cleared');
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PluginManager;
}

