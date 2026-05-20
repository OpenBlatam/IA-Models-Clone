/**
 * Offline Manager Module
 * =====================
 * Manages offline functionality, caching, and sync
 */

const OfflineManager = {
    /**
     * Offline status
     */
    isOnline: navigator.onLine,
    
    /**
     * Queue for offline requests
     */
    requestQueue: [],
    
    /**
     * Cache for offline data
     */
    offlineCache: new Map(),
    
    /**
     * Initialize offline manager
     */
    init() {
        // Listen to online/offline events
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.handleOnline();
        });
        
        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.handleOffline();
        });
        
        // Load queue from storage
        this.loadQueue();
        
        // Process queue if online
        if (this.isOnline) {
            this.processQueue();
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Offline manager initialized', { isOnline: this.isOnline });
        }
        
        // Emit status event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('offline:status', { isOnline: this.isOnline });
        }
    },
    
    /**
     * Handle online event
     */
    handleOnline() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Connection restored');
        }
        
        if (typeof Notifications !== 'undefined') {
            Notifications.success('Conexión restaurada');
        }
        
        // Process queued requests
        this.processQueue();
        
        // Emit event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('offline:online');
        }
    },
    
    /**
     * Handle offline event
     */
    handleOffline() {
        if (typeof Logger !== 'undefined') {
            Logger.warn('Connection lost');
        }
        
        if (typeof Notifications !== 'undefined') {
            Notifications.warning('Sin conexión. Los cambios se guardarán localmente.');
        }
        
        // Emit event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('offline:offline');
        }
    },
    
    /**
     * Check if online
     */
    checkOnline() {
        this.isOnline = navigator.onLine;
        return this.isOnline;
    },
    
    /**
     * Queue a request for later processing
     */
    queueRequest(request) {
        request.id = request.id || `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        request.timestamp = new Date().toISOString();
        request.retries = request.retries || 0;
        
        this.requestQueue.push(request);
        this.saveQueue();
        
        if (typeof Logger !== 'undefined') {
            Logger.debug('Request queued', { id: request.id, endpoint: request.endpoint });
        }
        
        return request.id;
    },
    
    /**
     * Process queued requests
     */
    async processQueue() {
        if (!this.isOnline || this.requestQueue.length === 0) {
            return;
        }
        
        const requests = [...this.requestQueue];
        this.requestQueue = [];
        
        for (const request of requests) {
            try {
                await this.executeRequest(request);
                if (typeof Logger !== 'undefined') {
                    Logger.info('Queued request processed', { id: request.id });
                }
            } catch (error) {
                // Re-queue if failed (with retry limit)
                if (request.retries < 3) {
                    request.retries++;
                    this.requestQueue.push(request);
                    if (typeof Logger !== 'undefined') {
                        Logger.warn('Request failed, re-queued', { id: request.id, retries: request.retries });
                    }
                } else {
                    if (typeof Logger !== 'undefined') {
                        Logger.error('Request failed after max retries', { id: request.id });
                    }
                }
            }
        }
        
        this.saveQueue();
    },
    
    /**
     * Execute a queued request
     */
    async executeRequest(request) {
        if (typeof API === 'undefined') {
            throw new Error('API module not available');
        }
        
        // Execute based on request type
        switch (request.type) {
            case 'change-clothing':
                return await API.changeClothing(request.data);
            case 'health':
                return await API.checkHealth();
            default:
                throw new Error(`Unknown request type: ${request.type}`);
        }
    },
    
    /**
     * Cache data for offline access
     */
    cacheData(key, data) {
        this.offlineCache.set(key, {
            data,
            timestamp: Date.now()
        });
        
        // Save to localStorage
        try {
            const cacheData = Array.from(this.offlineCache.entries()).map(([k, v]) => [k, v]);
            localStorage.setItem('offline_cache', JSON.stringify(cacheData));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save offline cache', error);
            }
        }
    },
    
    /**
     * Get cached data
     */
    getCachedData(key, maxAge = 3600000) { // 1 hour default
        const cached = this.offlineCache.get(key);
        if (!cached) return null;
        
        const age = Date.now() - cached.timestamp;
        if (age > maxAge) {
            this.offlineCache.delete(key);
            return null;
        }
        
        return cached.data;
    },
    
    /**
     * Save queue to storage
     */
    saveQueue() {
        try {
            localStorage.setItem('offline_queue', JSON.stringify(this.requestQueue));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save offline queue', error);
            }
        }
    },
    
    /**
     * Load queue from storage
     */
    loadQueue() {
        try {
            const stored = localStorage.getItem('offline_queue');
            if (stored) {
                this.requestQueue = JSON.parse(stored);
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to load offline queue', error);
            }
        }
    },
    
    /**
     * Clear queue
     */
    clearQueue() {
        this.requestQueue = [];
        this.saveQueue();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Offline queue cleared');
        }
    },
    
    /**
     * Get queue status
     */
    getQueueStatus() {
        return {
            isOnline: this.isOnline,
            queueLength: this.requestQueue.length,
            cacheSize: this.offlineCache.size
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OfflineManager;
}

