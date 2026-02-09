/**
 * API Performance Module
 * =====================
 * Handles performance tracking for API requests
 */

const APIPerformance = {
    /**
     * Track API call performance
     */
    track(endpoint, method, duration, success, status = null) {
        // Track with PerformanceMonitor
        if (typeof PerformanceMonitor !== 'undefined') {
            PerformanceMonitor.trackAPICall(endpoint, duration, success);
        }
        
        // Track with AdvancedAnalytics
        if (typeof AdvancedAnalytics !== 'undefined') {
            AdvancedAnalytics.trackAPICall(endpoint, method, duration, success, status);
        }
    },

    /**
     * Start performance measurement
     */
    start() {
        return performance.now();
    },

    /**
     * End performance measurement
     */
    end(startTime) {
        return performance.now() - startTime;
    },

    /**
     * Emit performance event
     */
    emitEvent(type, data) {
        if (typeof EventBus !== 'undefined') {
            EventBus.emit(`api:${type}`, data);
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APIPerformance;
}

