/**
 * Debug Tools Module
 * ==================
 * Development and debugging utilities
 */

const DebugTools = {
    /**
     * Debug mode
     */
    enabled: window.location.hostname === 'localhost' || 
             window.location.search.includes('debug=true'),
    
    /**
     * Debug panel element
     */
    panel: null,
    
    /**
     * Initialize debug tools
     */
    init() {
        if (!this.enabled) {
            return;
        }
        
        this.createDebugPanel();
        this.setupKeyboardShortcuts();
        this.logSystemInfo();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Debug tools initialized');
        }
    },
    
    /**
     * Create debug panel
     */
    createDebugPanel() {
        this.panel = document.createElement('div');
        this.panel.id = 'debugPanel';
        this.panel.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-height: 600px;
            background: rgba(0, 0, 0, 0.9);
            color: #0f0;
            font-family: monospace;
            font-size: 12px;
            padding: 15px;
            border-radius: 8px;
            z-index: 99999;
            display: none;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        `;
        
        this.panel.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #0f0;">🐛 Debug Panel</h3>
                <button id="debugClose" style="background: #f00; color: white; border: none; padding: 5px 10px; cursor: pointer;">✕</button>
            </div>
            <div id="debugContent"></div>
        `;
        
        document.body.appendChild(this.panel);
        
        // Close button
        document.getElementById('debugClose').addEventListener('click', () => {
            this.panel.style.display = 'none';
        });
        
        // Toggle on double-click
        this.panel.addEventListener('dblclick', () => {
            this.panel.style.display = this.panel.style.display === 'none' ? 'block' : 'none';
        });
    },
    
    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+D to toggle debug panel
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.togglePanel();
            }
            
            // Ctrl+Shift+L to show logs
            if (e.ctrlKey && e.shiftKey && e.key === 'L') {
                e.preventDefault();
                this.showLogs();
            }
            
            // Ctrl+Shift+S to show state
            if (e.ctrlKey && e.shiftKey && e.key === 'S') {
                e.preventDefault();
                this.showState();
            }
        });
    },
    
    /**
     * Toggle debug panel
     */
    togglePanel() {
        if (this.panel) {
            this.panel.style.display = this.panel.style.display === 'none' ? 'block' : 'none';
        }
    },
    
    /**
     * Show logs
     */
    showLogs() {
        if (!this.panel) return;
        
        const content = document.getElementById('debugContent');
        if (typeof Logger !== 'undefined') {
            const logs = Logger.getHistory();
            content.innerHTML = `
                <h4>Logs (${logs.length})</h4>
                <pre style="max-height: 400px; overflow-y: auto; background: #111; padding: 10px; border-radius: 4px;">
${logs.slice(-50).map(log => 
    `[${log.timestamp}] [${log.level}] ${log.message}${log.args ? ' ' + JSON.stringify(log.args) : ''}`
).join('\n')}
                </pre>
            `;
        } else {
            content.innerHTML = '<p>Logger not available</p>';
        }
        
        this.panel.style.display = 'block';
    },
    
    /**
     * Show state
     */
    showState() {
        if (!this.panel) return;
        
        const content = document.getElementById('debugContent');
        let stateInfo = '';
        
        if (typeof StateManager !== 'undefined') {
            stateInfo += `<h4>State Manager</h4><pre>${JSON.stringify(StateManager.getState(), null, 2)}</pre>`;
        }
        
        if (typeof EventBus !== 'undefined') {
            const events = EventBus.getEvents();
            stateInfo += `<h4>Event Bus (${events.length} events)</h4><pre>${events.join(', ')}</pre>`;
        }
        
        if (typeof Cache !== 'undefined') {
            const stats = Cache.getStats();
            stateInfo += `<h4>Cache Stats</h4><pre>${JSON.stringify(stats, null, 2)}</pre>`;
        }
        
        content.innerHTML = stateInfo || '<p>No state information available</p>';
        this.panel.style.display = 'block';
    },
    
    /**
     * Log system info
     */
    logSystemInfo() {
        if (typeof Logger === 'undefined') return;
        
        Logger.group('System Information', () => {
            Logger.info('User Agent:', navigator.userAgent);
            Logger.info('Platform:', navigator.platform);
            Logger.info('Language:', navigator.language);
            Logger.info('Screen:', `${screen.width}x${screen.height}`);
            Logger.info('Viewport:', `${window.innerWidth}x${window.innerHeight}`);
            
            if (performance.memory) {
                Logger.info('Memory:', {
                    used: `${(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
                    total: `${(performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
                    limit: `${(performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`
                });
            }
        });
    },
    
    /**
     * Export debug info
     */
    exportDebugInfo() {
        const info = {
            timestamp: new Date().toISOString(),
            system: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                language: navigator.language,
                screen: `${screen.width}x${screen.height}`,
                viewport: `${window.innerWidth}x${window.innerHeight}`
            },
            modules: typeof ModuleLoader !== 'undefined' 
                ? ModuleLoader.getLoadedModules() 
                : [],
            state: typeof StateManager !== 'undefined'
                ? StateManager.getState()
                : null,
            logs: typeof Logger !== 'undefined'
                ? Logger.getHistory()
                : [],
            performance: typeof PerformanceMonitor !== 'undefined'
                ? PerformanceMonitor.getStats()
                : null,
            health: typeof HealthMonitor !== 'undefined'
                ? HealthMonitor.getReport()
                : null
        };
        
        return JSON.stringify(info, null, 2);
    }
};

// Auto-initialize in debug mode
if (DebugTools.enabled && typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => DebugTools.init());
    } else {
        DebugTools.init();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DebugTools;
}

