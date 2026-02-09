/**
 * API Module
 * ==========
 * Handles all API communication with improved error handling
 */

const API = {
    /**
     * Base configuration
     */
    baseURL: CONFIG.API_BASE || 'http://localhost:8002/api/v1',
    
    /**
     * Make a fetch request with error handling, caching, and performance tracking
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const cacheKey = APICache.getCacheKey(endpoint, options);
        const shouldCache = APICache.shouldCache(options.method, options.useCache);
        
        // Check cache first
        if (shouldCache) {
            const cached = APICache.get(cacheKey);
            if (cached) {
                return cached;
            }
        }
        
        const defaultOptions = {
            headers: {
                'Accept': 'application/json',
            },
        };
        
        const mergedOptions = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...(options.headers || {}),
            },
        };
        
        // Track performance
        const startTime = APIPerformance.start();
        let success = false;
        
        try {
            // Emit request start event
            APIPerformance.emitEvent('request:start', { endpoint, options });
            
            const response = await fetch(url, mergedOptions);
            const data = await response.json();
            
            const duration = APIPerformance.end(startTime);
            success = response.ok;
            
            // Track performance
            APIPerformance.track(endpoint, options.method || 'GET', duration, success, response.status);
            
            if (!response.ok) {
                const errorResponse = {
                    success: false,
                    error: data.error || data.detail || `HTTP ${response.status}`,
                    status: response.status,
                    data: data,
                };
                
                // Emit error event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('api:error', { endpoint, error: errorResponse });
                }
                
                return errorResponse;
            }
            
            let responseData = data;
            
            // Execute api:after_request hook
            if (typeof PluginManager !== 'undefined') {
                responseData = PluginManager.executeHook('api:after_request', data, endpoint, method, response.status);
            } else if (typeof PluginSystem !== 'undefined') {
                responseData = PluginSystem.executeHook('api:after_request', data, endpoint, method, response.status);
            }
            
            const successResponse = {
                success: true,
                data: responseData,
                status: response.status,
            };
            
            // Cache successful GET requests
            if (shouldCache) {
                const ttl = options.cacheTTL || 5 * 60 * 1000; // 5 minutes default
                APICache.set(cacheKey, successResponse, ttl);
            }
            
            // Emit success event
            APIPerformance.emitEvent('success', { endpoint, data: successResponse, duration });
            
            return successResponse;
        } catch (error) {
            const duration = APIPerformance.end(startTime);
            
            // Track performance
            APIPerformance.track(endpoint, options.method || 'GET', duration, false, null);
            
            const errorResponse = {
                success: false,
                error: error.message || 'Network error',
                type: error.name || 'NetworkError',
            };
            
            // Emit error event
            APIPerformance.emitEvent('error', { endpoint, error: errorResponse });
            
            // Handle error
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handle(error, { context: 'API request', endpoint });
            }
            
            return errorResponse;
        }
    },
    
    /**
     * Check server health status
     */
    async checkHealth() {
        return this.request('/health');
    },
    
    /**
     * Change clothing on character
     */
    async changeClothing(formData) {
        return this.request('/change-clothing', {
            method: 'POST',
            body: formData,
        });
    },
    
    /**
     * Get model information
     */
    async getModelInfo() {
        return this.request('/model/info');
    },
    
    /**
     * Initialize model
     */
    async initializeModel() {
        return this.request('/initialize', {
            method: 'POST',
        });
    },
    
    /**
     * List tensors
     */
    async listTensors() {
        return this.request('/tensors');
    },
    
    /**
     * Get image by name
     */
    getImageUrl(imageName) {
        return `${this.baseURL}/image/${imageName}`;
    },
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}
