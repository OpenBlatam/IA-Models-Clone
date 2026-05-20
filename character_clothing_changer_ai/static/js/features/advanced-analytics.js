/**
 * Advanced Analytics Module
 * =========================
 * Tracks user behavior, performance metrics, and application analytics
 */

const AdvancedAnalytics = {
    /**
     * Analytics data storage
     */
    data: {
        events: [],
        metrics: {},
        sessions: [],
        userActions: [],
        performance: []
    },

    /**
     * Track an event
     */
    trackEvent(eventName, properties = {}) {
        const event = {
            name: eventName,
            properties,
            timestamp: new Date().toISOString(),
            sessionId: this.getSessionId()
        };

        this.data.events.push(event);

        // Keep only last 1000 events
        if (this.data.events.length > 1000) {
            this.data.events.shift();
        }

        // Emit analytics event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('analytics:event', event);
        }

        if (typeof Logger !== 'undefined') {
            Logger.debug(`Analytics Event: ${eventName}`, properties);
        }
    },

    /**
     * Track a metric
     */
    trackMetric(metricName, value, unit = '') {
        if (!this.data.metrics[metricName]) {
            this.data.metrics[metricName] = {
                values: [],
                unit,
                count: 0,
                sum: 0,
                min: Infinity,
                max: -Infinity,
                avg: 0
            };
        }

        const metric = this.data.metrics[metricName];
        metric.values.push(value);
        metric.count++;
        metric.sum += value;
        metric.min = Math.min(metric.min, value);
        metric.max = Math.max(metric.max, value);
        metric.avg = metric.sum / metric.count;

        // Keep only last 100 values per metric
        if (metric.values.length > 100) {
            metric.values.shift();
        }

        if (typeof EventBus !== 'undefined') {
            EventBus.emit('analytics:metric', { metricName, value, unit });
        }
    },

    /**
     * Track user action
     */
    trackUserAction(action, details = {}) {
        const userAction = {
            action,
            details,
            timestamp: new Date().toISOString(),
            sessionId: this.getSessionId(),
            url: window.location.href,
            userAgent: navigator.userAgent
        };

        this.data.userActions.push(userAction);

        // Keep only last 500 actions
        if (this.data.userActions.length > 500) {
            this.data.userActions.shift();
        }

        this.trackEvent('user_action', { action, ...details });
    },

    /**
     * Track performance metric
     */
    trackPerformance(metricName, duration, details = {}) {
        const performance = {
            metric: metricName,
            duration,
            details,
            timestamp: new Date().toISOString()
        };

        this.data.performance.push(performance);
        this.trackMetric(`performance.${metricName}`, duration, 'ms');

        // Keep only last 200 performance entries
        if (this.data.performance.length > 200) {
            this.data.performance.shift();
        }
    },

    /**
     * Track page view
     */
    trackPageView(pageName, properties = {}) {
        this.trackEvent('page_view', {
            page: pageName,
            ...properties
        });
    },

    /**
     * Track error
     */
    trackError(error, context = {}) {
        this.trackEvent('error', {
            error: error.message || error,
            stack: error.stack,
            context
        });

        this.trackMetric('errors.count', 1);
    },

    /**
     * Track API call
     */
    trackAPICall(endpoint, method, duration, success, statusCode = null) {
        this.trackEvent('api_call', {
            endpoint,
            method,
            duration,
            success,
            statusCode
        });

        this.trackMetric(`api.${endpoint}.duration`, duration, 'ms');
        this.trackMetric(`api.${endpoint}.${success ? 'success' : 'failure'}`, 1);
    },

    /**
     * Track form submission
     */
    trackFormSubmission(formName, success, duration = null) {
        this.trackEvent('form_submission', {
            form: formName,
            success,
            duration
        });

        if (duration !== null) {
            this.trackMetric(`form.${formName}.duration`, duration, 'ms');
        }
    },

    /**
     * Track image processing
     */
    trackImageProcessing(imageSize, processingTime, success) {
        this.trackEvent('image_processing', {
            imageSize,
            processingTime,
            success
        });

        this.trackMetric('image.processing_time', processingTime, 'ms');
        this.trackMetric('image.size', imageSize, 'bytes');
    },

    /**
     * Get session ID
     */
    getSessionId() {
        let sessionId = sessionStorage.getItem('analytics_session_id');
        if (!sessionId) {
            sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            sessionStorage.setItem('analytics_session_id', sessionId);
        }
        return sessionId;
    },

    /**
     * Get analytics summary
     */
    getSummary() {
        return {
            events: {
                total: this.data.events.length,
                byName: this.groupBy(this.data.events, 'name')
            },
            metrics: Object.keys(this.data.metrics).reduce((acc, key) => {
                const metric = this.data.metrics[key];
                acc[key] = {
                    count: metric.count,
                    avg: metric.avg,
                    min: metric.min,
                    max: metric.max,
                    unit: metric.unit
                };
                return acc;
            }, {}),
            userActions: {
                total: this.data.userActions.length,
                byAction: this.groupBy(this.data.userActions, 'action')
            },
            performance: {
                total: this.data.performance.length,
                avgDuration: this.data.performance.reduce((sum, p) => sum + p.duration, 0) / this.data.performance.length || 0
            }
        };
    },

    /**
     * Group array by property
     */
    groupBy(array, property) {
        return array.reduce((acc, item) => {
            const key = item[property];
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
    },

    /**
     * Export analytics data
     */
    export() {
        return {
            summary: this.getSummary(),
            events: this.data.events,
            metrics: this.data.metrics,
            userActions: this.data.userActions,
            performance: this.data.performance,
            exportedAt: new Date().toISOString()
        };
    },

    /**
     * Clear analytics data
     */
    clear() {
        this.data = {
            events: [],
            metrics: {},
            sessions: [],
            userActions: [],
            performance: []
        };

        if (typeof Logger !== 'undefined') {
            Logger.info('Analytics data cleared');
        }
    },

    /**
     * Initialize analytics
     */
    init() {
        // Track page load
        if (document.readyState === 'complete') {
            this.trackPageView(window.location.pathname);
        } else {
            window.addEventListener('load', () => {
                this.trackPageView(window.location.pathname);
            });
        }

        // Track errors
        window.addEventListener('error', (e) => {
            this.trackError(e.error || e.message, {
                filename: e.filename,
                lineno: e.lineno,
                colno: e.colno
            });
        });

        // Track unhandled promise rejections
        window.addEventListener('unhandledrejection', (e) => {
            this.trackError(e.reason, {
                type: 'unhandled_promise_rejection'
            });
        });

        // Track visibility changes
        document.addEventListener('visibilitychange', () => {
            this.trackEvent('visibility_change', {
                hidden: document.hidden
            });
        });

        // Track session start
        this.trackEvent('session_start', {
            sessionId: this.getSessionId(),
            referrer: document.referrer,
            url: window.location.href
        });

        if (typeof Logger !== 'undefined') {
            Logger.info('Advanced analytics initialized');
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => AdvancedAnalytics.init());
    } else {
        AdvancedAnalytics.init();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedAnalytics;
}
