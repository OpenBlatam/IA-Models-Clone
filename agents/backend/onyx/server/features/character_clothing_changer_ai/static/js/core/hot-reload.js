/**
 * Hot Reload Module
 * =================
 * Hot module reloading system for development
 */

const HotReload = {
    /**
     * Reloaded modules registry
     */
    reloadedModules: new Set(),
    
    /**
     * Module watchers
     */
    watchers: new Map(),
    
    /**
     * Enable hot reload
     */
    enabled: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1',
    
    /**
     * Initialize hot reload
     */
    init() {
        if (!this.enabled) {
            return;
        }
        
        // Setup WebSocket connection for file changes
        this.setupWebSocket();
        
        // Setup file watchers
        this.setupFileWatchers();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Hot reload enabled');
        }
    },
    
    /**
     * Setup WebSocket connection
     */
    setupWebSocket() {
        try {
            const ws = new WebSocket(`ws://${window.location.hostname}:8003`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'file-changed') {
                    this.reloadModule(data.file);
                }
            };
            
            ws.onerror = () => {
                // WebSocket not available, use polling instead
                this.setupPolling();
            };
            
            this.ws = ws;
        } catch (e) {
            // WebSocket not available, use polling
            this.setupPolling();
        }
    },
    
    /**
     * Setup polling for file changes
     */
    setupPolling() {
        setInterval(() => {
            this.checkForChanges();
        }, 2000); // Check every 2 seconds
    },
    
    /**
     * Setup file watchers
     */
    setupFileWatchers() {
        // Watch for module changes
        if (typeof EventBus !== 'undefined') {
            EventBus.on('module:loaded', (moduleName) => {
                this.watchModule(moduleName);
            });
        }
    },
    
    /**
     * Watch a module for changes
     */
    watchModule(moduleName) {
        if (this.watchers.has(moduleName)) {
            return;
        }
        
        const modulePath = this.getModulePath(moduleName);
        if (!modulePath) {
            return;
        }
        
        // Store module info
        this.watchers.set(moduleName, {
            path: modulePath,
            lastModified: Date.now()
        });
    },
    
    /**
     * Get module path
     */
    getModulePath(moduleName) {
        // Try to find module in different locations
        const locations = [
            `static/js/${moduleName.toLowerCase()}.js`,
            `static/js/core/${moduleName.toLowerCase()}.js`,
            `static/js/utils/${moduleName.toLowerCase()}.js`,
            `static/js/ui/${moduleName.toLowerCase()}.js`,
            `static/js/features/${moduleName.toLowerCase()}.js`,
            `static/js/renderers/${moduleName.toLowerCase()}.js`
        ];
        
        return locations[0]; // Simplified, would need actual file checking
    },
    
    /**
     * Check for file changes
     */
    async checkForChanges() {
        for (const [moduleName, watcher] of this.watchers.entries()) {
            try {
                const response = await fetch(watcher.path, { method: 'HEAD' });
                const lastModified = new Date(response.headers.get('Last-Modified')).getTime();
                
                if (lastModified > watcher.lastModified) {
                    this.reloadModule(moduleName);
                    watcher.lastModified = lastModified;
                }
            } catch (e) {
                // File not found or error, skip
            }
        }
    },
    
    /**
     * Reload a module
     */
    async reloadModule(moduleName) {
        if (this.reloadedModules.has(moduleName)) {
            return;
        }
        
        this.reloadedModules.add(moduleName);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Hot reloading module: ${moduleName}`);
        }
        
        // Emit reload event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('module:reload', moduleName);
        }
        
        // Reload module script
        const modulePath = this.getModulePath(moduleName);
        if (modulePath) {
            await this.loadModule(modulePath);
        }
        
        // Notify user
        if (typeof Notifications !== 'undefined') {
            Notifications.info(`Module ${moduleName} reloaded`, 2000);
        }
    },
    
    /**
     * Load module script
     */
    async loadModule(path) {
        return new Promise((resolve) => {
            const script = document.createElement('script');
            script.src = `${path}?t=${Date.now()}`; // Cache busting
            script.onload = () => resolve();
            script.onerror = () => resolve(); // Don't fail on error
            document.head.appendChild(script);
        });
    },
    
    /**
     * Disable hot reload
     */
    disable() {
        this.enabled = false;
        if (this.ws) {
            this.ws.close();
        }
    }
};

// Auto-initialize if enabled
if (typeof window !== 'undefined' && HotReload.enabled) {
    // Initialize after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => HotReload.init());
    } else {
        HotReload.init();
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HotReload;
}

