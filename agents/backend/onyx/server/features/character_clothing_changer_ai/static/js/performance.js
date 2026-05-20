/**
 * Performance Monitor Module
 * ==========================
 * Performance monitoring and metrics collection
 */

const PerformanceMonitor = {
    /**
     * Performance metrics
     */
    metrics: {
        apiCalls: [],
        renderTimes: [],
        memoryUsage: [],
        errors: []
    },

    /**
     * Initialize performance monitoring
     */
    init() {
        this.startMemoryMonitoring();
        this.setupErrorTracking();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Performance monitoring initialized');
        }
    },

    /**
     * Track API call performance
     */
    trackAPICall(endpoint, duration, success) {
        const metric = {
            endpoint,
            duration,
            success,
            timestamp: Date.now()
        };
        
        this.metrics.apiCalls.push(metric);
        
        // Keep only last 100 calls
        if (this.metrics.apiCalls.length > 100) {
            this.metrics.apiCalls.shift();
        }
        
        // Emit event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('performance:api', metric);
        }
        
        // Log slow calls
        if (duration > 5000 && typeof Logger !== 'undefined') {
            Logger.warn(`Slow API call: ${endpoint} took ${duration}ms`);
        }
    },

    /**
     * Track render time
     */
    trackRender(component, duration) {
        const metric = {
            component,
            duration,
            timestamp: Date.now()
        };
        
        this.metrics.renderTimes.push(metric);
        
        // Keep only last 50 renders
        if (this.metrics.renderTimes.length > 50) {
            this.metrics.renderTimes.shift();
        }
        
        // Emit event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('performance:render', metric);
        }
    },

    /**
     * Start performance measurement
     */
    startMeasure(label) {
        const startTime = performance.now();
        return {
            end: () => {
                const duration = performance.now() - startTime;
                if (typeof Logger !== 'undefined') {
                    Logger.debug(`${label} took ${duration.toFixed(2)}ms`);
                }
                return duration;
            }
        };
    },

    /**
     * Measure async function performance
     */
    async measureAsync(label, fn) {
        const measure = this.startMeasure(label);
        try {
            const result = await fn();
            const duration = measure.end();
            this.trackRender(label, duration);
            return result;
        } catch (error) {
            measure.end();
            throw error;
        }
    },

    /**
     * Start memory monitoring
     */
    startMemoryMonitoring() {
        if (!performance.memory) {
            return; // Not available in all browsers
        }
        
        setInterval(() => {
            const memory = {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit,
                timestamp: Date.now()
            };
            
            this.metrics.memoryUsage.push(memory);
            
            // Keep only last 100 measurements
            if (this.metrics.memoryUsage.length > 100) {
                this.metrics.memoryUsage.shift();
            }
            
            // Warn if memory usage is high
            const usagePercent = (memory.used / memory.limit) * 100;
            if (usagePercent > 80 && typeof Logger !== 'undefined') {
                Logger.warn(`High memory usage: ${usagePercent.toFixed(2)}%`);
            }
        }, 30000); // Every 30 seconds
    },

    /**
     * Setup error tracking
     */
    setupErrorTracking() {
        window.addEventListener('error', (event) => {
            this.metrics.errors.push({
                message: event.message,
                source: event.filename,
                line: event.lineno,
                column: event.colno,
                timestamp: Date.now()
            });
            
            // Keep only last 50 errors
            if (this.metrics.errors.length > 50) {
                this.metrics.errors.shift();
            }
        });
    },

    /**
     * Get performance statistics
     */
    getStats() {
        const apiStats = this.calculateAPIStats();
        const renderStats = this.calculateRenderStats();
        const memoryStats = this.calculateMemoryStats();
        
        return {
            api: apiStats,
            render: renderStats,
            memory: memoryStats,
            errors: this.metrics.errors.length
        };
    },

    /**
     * Calculate API statistics
     */
    calculateAPIStats() {
        if (this.metrics.apiCalls.length === 0) {
            return { count: 0, avgDuration: 0, successRate: 0 };
        }
        
        const durations = this.metrics.apiCalls.map(c => c.duration);
        const successes = this.metrics.apiCalls.filter(c => c.success).length;
        
        return {
            count: this.metrics.apiCalls.length,
            avgDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
            minDuration: Math.min(...durations),
            maxDuration: Math.max(...durations),
            successRate: (successes / this.metrics.apiCalls.length) * 100
        };
    },

    /**
     * Calculate render statistics
     */
    calculateRenderStats() {
        if (this.metrics.renderTimes.length === 0) {
            return { count: 0, avgDuration: 0 };
        }
        
        const durations = this.metrics.renderTimes.map(r => r.duration);
        
        return {
            count: this.metrics.renderTimes.length,
            avgDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
            minDuration: Math.min(...durations),
            maxDuration: Math.max(...durations)
        };
    },

    /**
     * Calculate memory statistics
     */
    calculateMemoryStats() {
        if (this.metrics.memoryUsage.length === 0) {
            return { current: 0, avg: 0, peak: 0 };
        }
        
        const current = this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1];
        const used = this.metrics.memoryUsage.map(m => m.used);
        
        return {
            current: current.used,
            avg: used.reduce((a, b) => a + b, 0) / used.length,
            peak: Math.max(...used),
            limit: current.limit
        };
    },

    /**
     * Clear all metrics
     */
    clear() {
        this.metrics = {
            apiCalls: [],
            renderTimes: [],
            memoryUsage: [],
            errors: []
        };
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Performance metrics cleared');
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceMonitor;
}

