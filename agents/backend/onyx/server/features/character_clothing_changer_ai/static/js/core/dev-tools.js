/**
 * Dev Tools Module
 * ================
 * Development tools and utilities
 */

const DevTools = {
    /**
     * Enabled flag
     */
    enabled: false,
    
    /**
     * Initialize dev tools
     */
    init() {
        // Check if dev mode
        this.enabled = localStorage.getItem('dev_mode') === 'true' || 
                      window.location.search.includes('dev=true');
        
        if (!this.enabled) {
            return;
        }
        
        // Setup dev tools
        this.setupConsole();
        this.setupShortcuts();
        this.setupPanel();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Dev tools initialized');
        }
    },
    
    /**
     * Setup enhanced console
     */
    setupConsole() {
        // Add custom console methods
        window.devConsole = {
            log: (...args) => {
                console.log('[DEV]', ...args);
            },
            state: () => {
                if (typeof StateManager !== 'undefined') {
                    console.log('State:', StateManager.getAll());
                }
            },
            cache: () => {
                if (typeof Cache !== 'undefined') {
                    console.log('Cache:', Cache.getAll());
                }
            },
            storage: () => {
                console.log('LocalStorage:', { ...localStorage });
            },
            modules: () => {
                const modules = {
                    StateManager: typeof StateManager !== 'undefined',
                    EventBus: typeof EventBus !== 'undefined',
                    Cache: typeof Cache !== 'undefined',
                    API: typeof API !== 'undefined',
                    Logger: typeof Logger !== 'undefined'
                };
                console.log('Modules:', modules);
            }
        };
    },
    
    /**
     * Setup keyboard shortcuts
     */
    setupShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+D: Toggle dev panel
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.togglePanel();
            }
            
            // Ctrl+Shift+C: Clear all data
            if (e.ctrlKey && e.shiftKey && e.key === 'C') {
                e.preventDefault();
                this.clearAllData();
            }
            
            // Ctrl+Shift+R: Reload modules
            if (e.ctrlKey && e.shiftKey && e.key === 'R') {
                e.preventDefault();
                this.reloadModules();
            }
        });
    },
    
    /**
     * Setup dev panel
     */
    setupPanel() {
        const panel = document.createElement('div');
        panel.id = 'dev-panel';
        panel.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.9);
            color: #0f0;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            display: none;
            max-width: 400px;
            max-height: 80vh;
            overflow-y: auto;
        `;
        
        panel.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <strong>DEV TOOLS</strong>
                <button id="dev-panel-close" style="background: #f00; color: #fff; border: none; padding: 2px 8px; cursor: pointer;">X</button>
            </div>
            <div id="dev-panel-content"></div>
        `;
        
        document.body.appendChild(panel);
        
        document.getElementById('dev-panel-close').addEventListener('click', () => {
            this.togglePanel();
        });
        
        this.updatePanel();
    },
    
    /**
     * Toggle dev panel
     */
    togglePanel() {
        const panel = document.getElementById('dev-panel');
        if (panel) {
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            if (panel.style.display === 'block') {
                this.updatePanel();
            }
        }
    },
    
    /**
     * Update dev panel
     */
    updatePanel() {
        const content = document.getElementById('dev-panel-content');
        if (!content) return;
        
        const info = {
            'State': typeof StateManager !== 'undefined' ? Object.keys(StateManager.getAll()).length : 0,
            'Cache': typeof Cache !== 'undefined' ? Cache.size() : 0,
            'Plugins': typeof PluginManager !== 'undefined' ? PluginManager.getAll().length : 0,
            'Components': typeof ComponentRegistry !== 'undefined' ? ComponentRegistry.getAll().length : 0,
            'Online': navigator.onLine,
            'Version': typeof VersionManager !== 'undefined' ? VersionManager.getCurrentVersion() : 'N/A'
        };
        
        content.innerHTML = Object.entries(info)
            .map(([key, value]) => `<div>${key}: ${value}</div>`)
            .join('');
    },
    
    /**
     * Clear all data
     */
    clearAllData() {
        if (confirm('Clear all application data?')) {
            localStorage.clear();
            if (typeof Cache !== 'undefined') {
                Cache.clear();
            }
            if (typeof StateManager !== 'undefined') {
                StateManager.clear();
            }
            location.reload();
        }
    },
    
    /**
     * Reload modules
     */
    reloadModules() {
        if (typeof HotReload !== 'undefined') {
            HotReload.reloadAll();
        } else {
            location.reload();
        }
    },
    
    /**
     * Export state
     */
    exportState() {
        const state = {
            state: typeof StateManager !== 'undefined' ? StateManager.getAll() : {},
            cache: typeof Cache !== 'undefined' ? Cache.getAll() : {},
            storage: { ...localStorage }
        };
        
        const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `dev-state-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }
};

// Auto-initialize if in dev mode
if (typeof window !== 'undefined') {
    DevTools.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DevTools;
}

