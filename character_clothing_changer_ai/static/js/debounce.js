/**
 * Debounce and Throttle Module
 * =============================
 * Utility functions for performance optimization
 */

const Debounce = {
    /**
     * Debounce function - delays execution until after wait time
     */
    debounce(func, wait, immediate = false) {
        let timeout;
        
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            
            const callNow = immediate && !timeout;
            
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            
            if (callNow) func(...args);
        };
    },

    /**
     * Throttle function - limits execution to once per wait time
     */
    throttle(func, wait) {
        let inThrottle;
        
        return function executedFunction(...args) {
            if (!inThrottle) {
                func(...args);
                inThrottle = true;
                setTimeout(() => {
                    inThrottle = false;
                }, wait);
            }
        };
    },

    /**
     * Create debounced version of a function
     */
    createDebounced(func, wait, immediate = false) {
        return this.debounce(func, wait, immediate);
    },

    /**
     * Create throttled version of a function
     */
    createThrottled(func, wait) {
        return this.throttle(func, wait);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Debounce;
}

