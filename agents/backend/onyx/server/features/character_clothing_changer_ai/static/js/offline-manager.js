/**
 * Offline Manager Module
 * ======================
 * Offline functionality and cache management
 */

const OfflineManager = {
    /**
     * Offline status
     */
    isOnline: navigator.onLine,
    
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
        
        // Initial status
        if (!this.isOnline) {
            this.handleOffline();
        }
        
        // Setup periodic connectivity check
        setInterval(() => {
            this.checkConnectivity();
        }, 30000); // Every 30 seconds
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Offline manager initialized');
        }
    },
    
    /**
     * Handle online event
     */
    handleOnline() {
        if (typeof Notifications !== 'undefined') {
            Notifications.success('Conexión restaurada');
        }
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('network:online');
        }
        
        // Sync pending operations
        this.syncPendingOperations();
    },
    
    /**
     * Handle offline event
     */
    handleOffline() {
        if (typeof Notifications !== 'undefined') {
            Notifications.warning('Sin conexión. Modo offline activado.');
        }
        
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('network:offline');
        }
    },
    
    /**
     * Check connectivity
     */
    async checkConnectivity() {
        try {
            const response = await fetch('/api/v1/health', {
                method: 'HEAD',
                cache: 'no-cache',
                timeout: 5000
            });
            
            const wasOffline = !this.isOnline;
            this.isOnline = response.ok;
            
            if (wasOffline && this.isOnline) {
                this.handleOnline();
            } else if (!wasOffline && !this.isOnline) {
                this.handleOffline();
            }
        } catch (error) {
            const wasOffline = !this.isOnline;
            this.isOnline = false;
            
            if (!wasOffline) {
                this.handleOffline();
            }
        }
    },
    
    /**
     * Sync pending operations
     */
    async syncPendingOperations() {
        // Get pending operations from storage
        if (typeof Storage !== 'undefined') {
            const pending = Storage.get('pending_operations', []);
            
            if (pending.length > 0) {
                if (typeof Logger !== 'undefined') {
                    Logger.info(`Syncing ${pending.length} pending operations`);
                }
                
                // Process pending operations
                for (const operation of pending) {
                    try {
                        await this.processPendingOperation(operation);
                    } catch (error) {
                        if (typeof Logger !== 'undefined') {
                            Logger.error('Failed to sync operation:', error);
                        }
                    }
                }
                
                // Clear pending operations
                Storage.remove('pending_operations');
            }
        }
    },
    
    /**
     * Process pending operation
     */
    async processPendingOperation(operation) {
        // This would be implemented based on operation type
        if (typeof API !== 'undefined' && operation.type === 'api_call') {
            await API.request(operation.endpoint, operation.options);
        }
    },
    
    /**
     * Queue operation for offline sync
     */
    queueOperation(operation) {
        if (typeof Storage !== 'undefined') {
            const pending = Storage.get('pending_operations', []);
            pending.push({
                ...operation,
                timestamp: Date.now()
            });
            Storage.save('pending_operations', pending);
        }
    },
    
    /**
     * Get offline status
     */
    getStatus() {
        return {
            isOnline: this.isOnline,
            timestamp: Date.now()
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OfflineManager;
}

