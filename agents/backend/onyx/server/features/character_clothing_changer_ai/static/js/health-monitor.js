/**
 * Health Monitor Module
 * ====================
 * System health monitoring and diagnostics
 */

const HealthMonitor = {
    /**
     * Health status
     */
    status: {
        healthy: true,
        issues: [],
        lastCheck: null,
        uptime: 0
    },
    
    /**
     * Initialize health monitor
     */
    init() {
        this.startTime = Date.now();
        this.startMonitoring();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Health monitor initialized');
        }
    },
    
    /**
     * Start monitoring
     */
    startMonitoring() {
        // Check health every 30 seconds
        setInterval(() => {
            this.checkHealth();
        }, 30000);
        
        // Initial check
        this.checkHealth();
    },
    
    /**
     * Check system health
     */
    checkHealth() {
        const issues = [];
        
        // Check localStorage
        try {
            localStorage.setItem('__health_check__', 'test');
            localStorage.removeItem('__health_check__');
        } catch (error) {
            issues.push({
                component: 'localStorage',
                severity: 'high',
                message: 'LocalStorage is not available',
                error: error.message
            });
        }
        
        // Check memory
        if (performance.memory) {
            const usage = (performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100;
            if (usage > 90) {
                issues.push({
                    component: 'memory',
                    severity: 'high',
                    message: `High memory usage: ${usage.toFixed(2)}%`
                });
            } else if (usage > 75) {
                issues.push({
                    component: 'memory',
                    severity: 'medium',
                    message: `Elevated memory usage: ${usage.toFixed(2)}%`
                });
            }
        }
        
        // Check API connectivity
        if (typeof API !== 'undefined') {
            API.checkHealth().then(response => {
                if (!response.success) {
                    issues.push({
                        component: 'api',
                        severity: 'high',
                        message: 'API server is not responding'
                    });
                }
            }).catch(error => {
                issues.push({
                    component: 'api',
                    severity: 'high',
                    message: 'API connection error',
                    error: error.message
                });
            });
        }
        
        // Check critical modules
        const criticalModules = ['EventBus', 'StateManager', 'ErrorHandler'];
        criticalModules.forEach(module => {
            if (typeof window[module] === 'undefined') {
                issues.push({
                    component: 'module',
                    severity: 'critical',
                    message: `Critical module missing: ${module}`
                });
            }
        });
        
        // Update status
        this.status = {
            healthy: issues.filter(i => i.severity === 'critical').length === 0,
            issues,
            lastCheck: Date.now(),
            uptime: Date.now() - this.startTime
        };
        
        // Emit health status event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('health:status', this.status);
        }
        
        // Log issues
        if (issues.length > 0 && typeof Logger !== 'undefined') {
            issues.forEach(issue => {
                if (issue.severity === 'critical') {
                    Logger.error(`Health issue: ${issue.message}`);
                } else if (issue.severity === 'high') {
                    Logger.warn(`Health issue: ${issue.message}`);
                } else {
                    Logger.info(`Health issue: ${issue.message}`);
                }
            });
        }
        
        return this.status;
    },
    
    /**
     * Get health status
     */
    getStatus() {
        return { ...this.status };
    },
    
    /**
     * Get health report
     */
    getReport() {
        const report = {
            ...this.status,
            timestamp: new Date().toISOString(),
            modules: typeof ModuleLoader !== 'undefined' 
                ? ModuleLoader.getLoadedModules() 
                : [],
            performance: typeof PerformanceMonitor !== 'undefined'
                ? PerformanceMonitor.getStats()
                : null,
            analytics: typeof Analytics !== 'undefined'
                ? Analytics.getSummary()
                : null
        };
        
        return report;
    },
    
    /**
     * Export health report
     */
    exportReport() {
        return JSON.stringify(this.getReport(), null, 2);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HealthMonitor;
}

