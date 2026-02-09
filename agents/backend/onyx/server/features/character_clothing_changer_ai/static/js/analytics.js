/**
 * Analytics Module
 * ================
 * User behavior tracking and analytics
 */

const Analytics = {
    /**
     * Analytics data
     */
    data: {
        events: [],
        pageViews: 0,
        startTime: Date.now()
    },

    /**
     * Initialize analytics
     */
    init() {
        this.trackPageView();
        this.setupAutoTracking();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Analytics initialized');
        }
    },

    /**
     * Track page view
     */
    trackPageView() {
        this.data.pageViews++;
        this.trackEvent('page_view', {
            timestamp: Date.now(),
            url: window.location.href
        });
    },

    /**
     * Track custom event
     */
    trackEvent(eventName, properties = {}) {
        const event = {
            name: eventName,
            properties,
            timestamp: Date.now()
        };
        
        this.data.events.push(event);
        
        // Keep only last 100 events
        if (this.data.events.length > 100) {
            this.data.events.shift();
        }
        
        // Emit analytics event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('analytics:event', event);
        }
        
        // Log in debug mode
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Analytics event: ${eventName}`, properties);
        }
    },

    /**
     * Track user action
     */
    trackAction(action, details = {}) {
        this.trackEvent('user_action', {
            action,
            ...details
        });
    },

    /**
     * Track form submission
     */
    trackFormSubmit(success, duration = null) {
        this.trackEvent('form_submit', {
            success,
            duration
        });
    },

    /**
     * Track image upload
     */
    trackImageUpload(fileSize, fileType) {
        this.trackEvent('image_upload', {
            fileSize,
            fileType
        });
    },

    /**
     * Track tab switch
     */
    trackTabSwitch(tabName) {
        this.trackEvent('tab_switch', {
            tab: tabName
        });
    },

    /**
     * Setup automatic tracking
     */
    setupAutoTracking() {
        // Track form submissions
        if (typeof EventBus !== 'undefined') {
            EventBus.on('form:submitted', () => {
                this.trackAction('form_submit_start');
            });
            
            EventBus.on('form:completed', () => {
                this.trackAction('form_submit_success');
            });
            
            EventBus.on('form:error', () => {
                this.trackAction('form_submit_error');
            });
            
            // Track tab switches
            EventBus.on('tab:changed', (tab) => {
                this.trackTabSwitch(tab);
            });
            
            // Track API calls
            EventBus.on('api:success', (data) => {
                this.trackEvent('api_success', {
                    endpoint: data.endpoint,
                    duration: data.duration
                });
            });
            
            EventBus.on('api:error', (data) => {
                this.trackEvent('api_error', {
                    endpoint: data.endpoint
                });
            });
        }
    },

    /**
     * Get analytics summary
     */
    getSummary() {
        const sessionDuration = Date.now() - this.data.startTime;
        const eventsByType = {};
        
        this.data.events.forEach(event => {
            eventsByType[event.name] = (eventsByType[event.name] || 0) + 1;
        });
        
        return {
            pageViews: this.data.pageViews,
            totalEvents: this.data.events.length,
            eventsByType,
            sessionDuration: Math.floor(sessionDuration / 1000), // seconds
            startTime: this.data.startTime
        };
    },

    /**
     * Export analytics data
     */
    export() {
        return JSON.stringify({
            summary: this.getSummary(),
            events: this.data.events
        }, null, 2);
    },

    /**
     * Clear analytics data
     */
    clear() {
        this.data = {
            events: [],
            pageViews: 0,
            startTime: Date.now()
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Analytics;
}

