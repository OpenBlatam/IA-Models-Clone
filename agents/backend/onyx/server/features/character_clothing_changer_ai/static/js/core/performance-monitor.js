/**
 * Performance Monitor Module
 * ==========================
 * Advanced performance monitoring and metrics collection
 */

const PerformanceMonitor = {
    /**
     * Metrics storage
     */
    metrics: {
        pageLoad: null,
        apiCalls: [],
        renderTimes: [],
        memoryUsage: [],
        errors: []
    },
    
    /**
     * Observers
     */
    observers: {
        performance: null,
        memory: null
    },
    
    /**
     * Initialize performance monitor
     */
    init() {
        // Measure page load time
        if (window.performance && window.performance.timing) {
            window.addEventListener('load', () => {
                this.measurePageLoad();
            });
        }
        
        // Setup performance observer
        this.setupPerformanceObserver();
        
        // Setup memory monitoring
        this.setupMemoryMonitoring();
        
        // Monitor API calls
        this.monitorAPICalls();
        
        // Monitor render times
        this.monitorRenderTimes();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Performance monitor initialized');
        }
    },
    
    /**
     * Measure page load time
     */
    measurePageLoad() {
        if (!window.performance || !window.performance.timing) {
            return;
        }
        
        const timing = window.performance.timing;
        const loadTime = timing.loadEventEnd - timing.navigationStart;
        const domReady = timing.domContentLoadedEventEnd - timing.navigationStart;
        const firstPaint = timing.responseEnd - timing.requestStart;
        
        this.metrics.pageLoad = {
            total: loadTime,
            domReady,
            firstPaint,
            dns: timing.domainLookupEnd - timing.domainLookupStart,
            tcp: timing.connectEnd - timing.connectStart,
            request: timing.responseStart - timing.requestStart,
            response: timing.responseEnd - timing.responseStart,
            processing: timing.domComplete - timing.domLoading,
            timestamp: Date.now()
        };
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('performance:page_load', this.metrics.pageLoad);
        }
    },
    
    /**
     * Setup performance observer
     */
    setupPerformanceObserver() {
        if (typeof PerformanceObserver === 'undefined') {
            return;
        }
        
        try {
            this.observers.performance = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                entries.forEach(entry => {
                    this.handlePerformanceEntry(entry);
                });
            });
            
            this.observers.performance.observe({ 
                entryTypes: ['measure', 'mark', 'navigation', 'resource'] 
            });
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Performance Observer not supported', error);
            }
        }
    },
    
    /**
     * Handle performance entry
     */
    handlePerformanceEntry(entry) {
        switch (entry.entryType) {
            case 'measure':
                this.metrics.renderTimes.push({
                    name: entry.name,
                    duration: entry.duration,
                    timestamp: entry.startTime
                });
                break;
            case 'resource':
                if (entry.name.includes('/api/')) {
                    this.metrics.apiCalls.push({
                        url: entry.name,
                        duration: entry.duration,
                        size: entry.transferSize,
                        timestamp: entry.startTime
                    });
                }
                break;
        }
    },
    
    /**
     * Setup memory monitoring
     */
    setupMemoryMonitoring() {
        if (!performance.memory) {
            return;
        }
        
        setInterval(() => {
            const memory = {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit,
                timestamp: Date.now()
            };
            
            this.metrics.memoryUsage.push(memory);
            
            // Keep only last 100 entries
            if (this.metrics.memoryUsage.length > 100) {
                this.metrics.memoryUsage.shift();
            }
            
            // Emit memory event if threshold exceeded
            const usagePercent = (memory.used / memory.limit) * 100;
            if (usagePercent > 80) {
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('performance:memory_warning', memory);
                }
            }
        }, 5000); // Every 5 seconds
    },
    
    /**
     * Monitor API calls
     */
    monitorAPICalls() {
        if (typeof EventBus === 'undefined') {
            return;
        }
        
        EventBus.on('api:request:start', (data) => {
            const startTime = performance.now();
            data._perfStart = startTime;
        });
        
        EventBus.on('api:success', (data) => {
            if (data._perfStart) {
                const duration = performance.now() - data._perfStart;
                this.metrics.apiCalls.push({
                    endpoint: data.endpoint,
                    duration,
                    success: true,
                    timestamp: Date.now()
                });
            }
        });
        
        EventBus.on('api:error', (data) => {
            if (data._perfStart) {
                const duration = performance.now() - data._perfStart;
                this.metrics.apiCalls.push({
                    endpoint: data.endpoint,
                    duration,
                    success: false,
                    error: data.error,
                    timestamp: Date.now()
                });
            }
        });
    },
    
    /**
     * Monitor render times
     */
    monitorRenderTimes() {
        if (typeof EventBus === 'undefined') {
            return;
        }
        
        EventBus.on('render:start', (data) => {
            const startTime = performance.now();
            data._renderStart = startTime;
        });
        
        EventBus.on('render:complete', (data) => {
            if (data._renderStart) {
                const duration = performance.now() - data._renderStart;
                this.metrics.renderTimes.push({
                    component: data.component,
                    duration,
                    timestamp: Date.now()
                });
            }
        });
    },
    
    /**
     * Mark performance
     */
    mark(name) {
        if (window.performance && window.performance.mark) {
            window.performance.mark(name);
        }
    },
    
    /**
     * Measure performance
     */
    measure(name, startMark, endMark) {
        if (window.performance && window.performance.measure) {
            try {
                window.performance.measure(name, startMark, endMark);
                const measure = window.performance.getEntriesByName(name, 'measure')[0];
                return measure ? measure.duration : null;
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.warn('Performance measure failed', error);
                }
                return null;
            }
        }
        return null;
    },
    
    /**
     * Get metrics
     */
    getMetrics() {
        return {
            pageLoad: this.metrics.pageLoad,
            apiCalls: {
                total: this.metrics.apiCalls.length,
                average: this.getAverageAPIDuration(),
                success: this.metrics.apiCalls.filter(c => c.success).length,
                failed: this.metrics.apiCalls.filter(c => !c.success).length,
                recent: this.metrics.apiCalls.slice(-10)
            },
            renderTimes: {
                total: this.metrics.renderTimes.length,
                average: this.getAverageRenderTime(),
                recent: this.metrics.renderTimes.slice(-10)
            },
            memory: {
                current: this.getCurrentMemory(),
                average: this.getAverageMemory(),
                peak: this.getPeakMemory()
            }
        };
    },
    
    /**
     * Get average API duration
     */
    getAverageAPIDuration() {
        if (this.metrics.apiCalls.length === 0) {
            return 0;
        }
        const total = this.metrics.apiCalls.reduce((sum, call) => sum + call.duration, 0);
        return total / this.metrics.apiCalls.length;
    },
    
    /**
     * Get average render time
     */
    getAverageRenderTime() {
        if (this.metrics.renderTimes.length === 0) {
            return 0;
        }
        const total = this.metrics.renderTimes.reduce((sum, time) => sum + time.duration, 0);
        return total / this.metrics.renderTimes.length;
    },
    
    /**
     * Get current memory
     */
    getCurrentMemory() {
        if (!performance.memory) {
            return null;
        }
        return {
            used: performance.memory.usedJSHeapSize,
            total: performance.memory.totalJSHeapSize,
            limit: performance.memory.jsHeapSizeLimit
        };
    },
    
    /**
     * Get average memory
     */
    getAverageMemory() {
        if (this.metrics.memoryUsage.length === 0) {
            return null;
        }
        const avg = this.metrics.memoryUsage.reduce((sum, mem) => sum + mem.used, 0) / this.metrics.memoryUsage.length;
        return avg;
    },
    
    /**
     * Get peak memory
     */
    getPeakMemory() {
        if (this.metrics.memoryUsage.length === 0) {
            return null;
        }
        return Math.max(...this.metrics.memoryUsage.map(mem => mem.used));
    },
    
    /**
     * Clear metrics
     */
    clearMetrics() {
        this.metrics = {
            pageLoad: null,
            apiCalls: [],
            renderTimes: [],
            memoryUsage: [],
            errors: []
        };
    },
    
    /**
     * Export metrics
     */
    exportMetrics() {
        const data = {
            metrics: this.getMetrics(),
            raw: this.metrics,
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `performance-metrics-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => PerformanceMonitor.init());
    } else {
        PerformanceMonitor.init();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceMonitor;
}

