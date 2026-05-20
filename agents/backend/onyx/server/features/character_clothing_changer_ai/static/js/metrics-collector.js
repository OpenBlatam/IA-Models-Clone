/**
 * Metrics Collector Module
 * ========================
 * Advanced metrics collection and analysis
 */

const MetricsCollector = {
    /**
     * Collected metrics
     */
    metrics: {
        userActions: [],
        performance: [],
        errors: [],
        apiCalls: [],
        uiInteractions: []
    },
    
    /**
     * Initialize metrics collector
     */
    init() {
        this.startCollection();
        this.setupEventListeners();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Metrics collector initialized');
        }
    },
    
    /**
     * Start collecting metrics
     */
    startCollection() {
        // Collect metrics every 30 seconds
        setInterval(() => {
            this.collectSystemMetrics();
        }, 30000);
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        if (typeof EventBus === 'undefined') return;
        
        // Track user actions
        EventBus.on('form:submitted', () => {
            this.recordUserAction('form_submit');
        });
        
        EventBus.on('tab:changed', (tab) => {
            this.recordUserAction('tab_switch', { tab });
        });
        
        // Track performance
        EventBus.on('performance:api', (data) => {
            this.recordPerformance('api_call', data);
        });
        
        EventBus.on('performance:render', (data) => {
            this.recordPerformance('render', data);
        });
        
        // Track errors
        EventBus.on('api:error', (data) => {
            this.recordError('api_error', data);
        });
    },
    
    /**
     * Record user action
     */
    recordUserAction(action, details = {}) {
        const metric = {
            action,
            details,
            timestamp: Date.now()
        };
        
        this.metrics.userActions.push(metric);
        
        // Keep only last 1000 actions
        if (this.metrics.userActions.length > 1000) {
            this.metrics.userActions.shift();
        }
        
        // Emit metric event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('metrics:user_action', metric);
        }
    },
    
    /**
     * Record performance metric
     */
    recordPerformance(type, data) {
        const metric = {
            type,
            ...data,
            timestamp: Date.now()
        };
        
        this.metrics.performance.push(metric);
        
        // Keep only last 500 performance metrics
        if (this.metrics.performance.length > 500) {
            this.metrics.performance.shift();
        }
    },
    
    /**
     * Record error
     */
    recordError(type, error) {
        const metric = {
            type,
            error,
            timestamp: Date.now()
        };
        
        this.metrics.errors.push(metric);
        
        // Keep only last 200 errors
        if (this.metrics.errors.length > 200) {
            this.metrics.errors.shift();
        }
    },
    
    /**
     * Collect system metrics
     */
    collectSystemMetrics() {
        const metrics = {
            timestamp: Date.now(),
            memory: performance.memory ? {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit
            } : null,
            navigation: {
                type: performance.navigation.type,
                redirectCount: performance.navigation.redirectCount
            },
            timing: performance.timing ? {
                domContentLoaded: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                loadComplete: performance.timing.loadEventEnd - performance.timing.navigationStart
            } : null
        };
        
        this.metrics.performance.push(metrics);
    },
    
    /**
     * Get metrics summary
     */
    getSummary() {
        return {
            userActions: {
                total: this.metrics.userActions.length,
                byAction: this.groupBy(this.metrics.userActions, 'action')
            },
            performance: {
                total: this.metrics.performance.length,
                average: this.calculateAverage(this.metrics.performance, 'duration')
            },
            errors: {
                total: this.metrics.errors.length,
                byType: this.groupBy(this.metrics.errors, 'type')
            }
        };
    },
    
    /**
     * Group metrics by field
     */
    groupBy(array, field) {
        return array.reduce((acc, item) => {
            const key = item[field];
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
    },
    
    /**
     * Calculate average
     */
    calculateAverage(array, field) {
        if (array.length === 0) return 0;
        const sum = array.reduce((acc, item) => acc + (item[field] || 0), 0);
        return sum / array.length;
    },
    
    /**
     * Export metrics
     */
    export() {
        return JSON.stringify({
            summary: this.getSummary(),
            raw: this.metrics,
            timestamp: Date.now()
        }, null, 2);
    },
    
    /**
     * Clear metrics
     */
    clear() {
        this.metrics = {
            userActions: [],
            performance: [],
            errors: [],
            apiCalls: [],
            uiInteractions: []
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetricsCollector;
}

