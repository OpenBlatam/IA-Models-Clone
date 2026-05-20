/**
 * Feature Flags Module
 * ===================
 * Feature toggle system
 */

const FeatureFlags = {
    /**
     * Feature flags
     */
    flags: {},
    
    /**
     * Initialize feature flags
     */
    init(defaultFlags = {}) {
        // Load from storage
        if (typeof Storage !== 'undefined') {
            const saved = Storage.get('feature_flags');
            if (saved) {
                this.flags = { ...defaultFlags, ...saved };
            } else {
                this.flags = { ...defaultFlags };
            }
        } else {
            this.flags = { ...defaultFlags };
        }
        
        // Default flags
        this.flags = {
            enableAnalytics: true,
            enablePerformanceMonitoring: true,
            enableOfflineMode: true,
            enableServiceWorker: true,
            enableDebugTools: window.location.hostname === 'localhost',
            ...this.flags
        };
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Feature flags initialized');
        }
    },
    
    /**
     * Check if feature is enabled
     */
    isEnabled(feature) {
        return this.flags[feature] === true;
    },
    
    /**
     * Enable feature
     */
    enable(feature) {
        this.flags[feature] = true;
        this.save();
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('feature:enabled', feature);
        }
    },
    
    /**
     * Disable feature
     */
    disable(feature) {
        this.flags[feature] = false;
        this.save();
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('feature:disabled', feature);
        }
    },
    
    /**
     * Toggle feature
     */
    toggle(feature) {
        if (this.isEnabled(feature)) {
            this.disable(feature);
        } else {
            this.enable(feature);
        }
    },
    
    /**
     * Set feature value
     */
    set(feature, value) {
        this.flags[feature] = value;
        this.save();
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('feature:changed', { feature, value });
        }
    },
    
    /**
     * Get all flags
     */
    getAll() {
        return { ...this.flags };
    },
    
    /**
     * Save flags to storage
     */
    save() {
        if (typeof Storage !== 'undefined') {
            Storage.save('feature_flags', this.flags);
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FeatureFlags;
}

