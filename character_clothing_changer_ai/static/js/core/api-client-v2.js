/**
 * API Client V2 Module
 * ===================
 * Advanced API client with retry, caching, and interceptors
 */

const ApiClientV2 = {
    /**
     * Base URL
     */
    baseURL: '',
    
    /**
     * Default headers
     */
    defaultHeaders: {
        'Content-Type': 'application/json'
    },
    
    /**
     * Request interceptors
     */
    requestInterceptors: [],
    
    /**
     * Response interceptors
     */
    responseInterceptors: [],
    
    /**
     * Retry configuration
     */
    retryConfig: {
        maxRetries: 3,
        retryDelay: 1000,
        retryableStatuses: [408, 429, 500, 502, 503, 504]
    },
    
    /**
     * Initialize API client
     */
    init(config = {}) {
        this.baseURL = config.baseURL || '';
        this.defaultHeaders = { ...this.defaultHeaders, ...config.headers };
        this.retryConfig = { ...this.retryConfig, ...config.retry };
        
        if (typeof Logger !== 'undefined') {
            Logger.info('API client V2 initialized', { baseURL: this.baseURL });
        }
    },
    
    /**
     * Add request interceptor
     */
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    },
    
    /**
     * Add response interceptor
     */
    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    },
    
    /**
     * Execute request interceptors
     */
    async executeRequestInterceptors(config) {
        let processedConfig = config;
        
        for (const interceptor of this.requestInterceptors) {
            processedConfig = await interceptor(processedConfig);
        }
        
        return processedConfig;
    },
    
    /**
     * Execute response interceptors
     */
    async executeResponseInterceptors(response) {
        let processedResponse = response;
        
        for (const interceptor of this.responseInterceptors) {
            processedResponse = await interceptor(processedResponse);
        }
        
        return processedResponse;
    },
    
    /**
     * Request with retry
     */
    async request(url, options = {}, retryCount = 0) {
        try {
            // Build full URL
            const fullURL = url.startsWith('http') ? url : `${this.baseURL}${url}`;
            
            // Merge headers
            const headers = {
                ...this.defaultHeaders,
                ...options.headers
            };
            
            // Build config
            let config = {
                ...options,
                url: fullURL,
                headers
            };
            
            // Execute request interceptors
            config = await this.executeRequestInterceptors(config);
            
            // Make request
            const response = await fetch(config.url, {
                method: config.method || 'GET',
                headers: config.headers,
                body: config.body ? JSON.stringify(config.body) : undefined
            });
            
            // Process response
            let processedResponse = {
                ok: response.ok,
                status: response.status,
                statusText: response.statusText,
                headers: response.headers,
                data: null
            };
            
            // Parse response
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                processedResponse.data = await response.json();
            } else {
                processedResponse.data = await response.text();
            }
            
            // Execute response interceptors
            processedResponse = await this.executeResponseInterceptors(processedResponse);
            
            // Retry on failure
            if (!response.ok && this.shouldRetry(response.status, retryCount)) {
                return this.retryRequest(url, options, retryCount);
            }
            
            // Emit API call event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('api:call', { url: fullURL, status: response.status });
            }
            
            return processedResponse;
            
        } catch (error) {
            // Retry on error
            if (this.shouldRetry(null, retryCount)) {
                return this.retryRequest(url, options, retryCount);
            }
            
            throw error;
        }
    },
    
    /**
     * Retry request
     */
    async retryRequest(url, options, retryCount) {
        const delay = this.retryConfig.retryDelay * Math.pow(2, retryCount);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Retrying request: ${url} (attempt ${retryCount + 1})`);
        }
        
        await new Promise(resolve => setTimeout(resolve, delay));
        
        return this.request(url, options, retryCount + 1);
    },
    
    /**
     * Should retry
     */
    shouldRetry(status, retryCount) {
        if (retryCount >= this.retryConfig.maxRetries) {
            return false;
        }
        
        if (status === null) {
            return true; // Network error
        }
        
        return this.retryConfig.retryableStatuses.includes(status);
    },
    
    /**
     * GET request
     */
    async get(url, options = {}) {
        return this.request(url, { ...options, method: 'GET' });
    },
    
    /**
     * POST request
     */
    async post(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'POST',
            body: data
        });
    },
    
    /**
     * PUT request
     */
    async put(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'PUT',
            body: data
        });
    },
    
    /**
     * DELETE request
     */
    async delete(url, options = {}) {
        return this.request(url, { ...options, method: 'DELETE' });
    },
    
    /**
     * PATCH request
     */
    async patch(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'PATCH',
            body: data
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ApiClientV2;
}

