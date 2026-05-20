/**
 * URL Utils Module
 * ================
 * Utilities for URL manipulation and parsing
 */

const URLUtils = {
    /**
     * Parse query parameters
     */
    parseQuery(queryString = window.location.search) {
        const params = {};
        const searchParams = new URLSearchParams(queryString);
        
        for (const [key, value] of searchParams.entries()) {
            params[key] = value;
        }
        
        return params;
    },
    
    /**
     * Build query string
     */
    buildQuery(params) {
        const searchParams = new URLSearchParams();
        
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                searchParams.append(key, params[key]);
            }
        });
        
        return searchParams.toString();
    },
    
    /**
     * Get query parameter
     */
    getQueryParam(key, defaultValue = null) {
        const params = this.parseQuery();
        return params[key] !== undefined ? params[key] : defaultValue;
    },
    
    /**
     * Set query parameter
     */
    setQueryParam(key, value, replace = true) {
        const url = new URL(window.location.href);
        
        if (value === null || value === undefined) {
            url.searchParams.delete(key);
        } else {
            url.searchParams.set(key, value);
        }
        
        if (replace) {
            window.history.replaceState({}, '', url);
        } else {
            window.history.pushState({}, '', url);
        }
    },
    
    /**
     * Remove query parameter
     */
    removeQueryParam(key) {
        this.setQueryParam(key, null);
    },
    
    /**
     * Get current path
     */
    getPath() {
        return window.location.pathname;
    },
    
    /**
     * Navigate to URL
     */
    navigate(url, replace = false) {
        if (replace) {
            window.location.replace(url);
        } else {
            window.location.href = url;
        }
    },
    
    /**
     * Check if URL is external
     */
    isExternal(url) {
        try {
            const urlObj = new URL(url, window.location.href);
            return urlObj.origin !== window.location.origin;
        } catch {
            return false;
        }
    },
    
    /**
     * Validate URL
     */
    isValidURL(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    },
    
    /**
     * Get base URL
     */
    getBaseURL() {
        return `${window.location.protocol}//${window.location.host}`;
    },
    
    /**
     * Get relative URL
     */
    getRelativeURL(url) {
        try {
            const urlObj = new URL(url, window.location.href);
            return urlObj.pathname + urlObj.search + urlObj.hash;
        } catch {
            return url;
        }
    },
    
    /**
     * Encode URL component
     */
    encode(component) {
        return encodeURIComponent(component);
    },
    
    /**
     * Decode URL component
     */
    decode(component) {
        return decodeURIComponent(component);
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = URLUtils;
}

