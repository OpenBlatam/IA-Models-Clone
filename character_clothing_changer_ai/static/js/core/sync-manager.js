/**
 * Sync Manager Module
 * ===================
 * Manages data synchronization between local storage and server
 */

const SyncManager = {
    /**
     * Sync queue
     */
    syncQueue: [],
    
    /**
     * Sync status
     */
    isSyncing: false,
    
    /**
     * Last sync timestamp
     */
    lastSync: null,
    
    /**
     * Sync interval (ms)
     */
    syncInterval: 60000, // 1 minute
    
    /**
     * Sync timer
     */
    syncTimer: null,
    
    /**
     * Initialize sync manager
     */
    init() {
        // Load sync queue from storage
        this.loadQueue();
        
        // Load last sync timestamp
        const stored = localStorage.getItem('last_sync');
        if (stored) {
            this.lastSync = new Date(stored);
        }
        
        // Start periodic sync
        this.startPeriodicSync();
        
        // Listen to online/offline events
        if (typeof OfflineManager !== 'undefined') {
            if (typeof EventBus !== 'undefined') {
                EventBus.on('offline:online', () => {
                    this.sync();
                });
            }
        }
        
        // Listen to data changes
        if (typeof EventBus !== 'undefined') {
            EventBus.on('data:changed', (data) => {
                this.queueSync(data);
            });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Sync manager initialized');
        }
    },
    
    /**
     * Start periodic sync
     */
    startPeriodicSync() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
        }
        
        this.syncTimer = setInterval(() => {
            if (!this.isSyncing && typeof OfflineManager !== 'undefined' && OfflineManager.isOnline) {
                this.sync();
            }
        }, this.syncInterval);
    },
    
    /**
     * Stop periodic sync
     */
    stopPeriodicSync() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
            this.syncTimer = null;
        }
    },
    
    /**
     * Queue data for sync
     */
    queueSync(data) {
        const syncItem = {
            id: `sync_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            data,
            timestamp: new Date().toISOString(),
            retries: 0
        };
        
        this.syncQueue.push(syncItem);
        this.saveQueue();
        
        if (typeof Logger !== 'undefined') {
            Logger.debug('Data queued for sync', { id: syncItem.id, type: data.type });
        }
        
        // Try to sync immediately if online
        if (typeof OfflineManager !== 'undefined' && OfflineManager.isOnline) {
            this.sync();
        }
    },
    
    /**
     * Sync all queued data
     */
    async sync() {
        if (this.isSyncing) {
            return;
        }
        
        if (typeof OfflineManager !== 'undefined' && !OfflineManager.isOnline) {
            if (typeof Logger !== 'undefined') {
                Logger.debug('Sync skipped - offline');
            }
            return;
        }
        
        if (this.syncQueue.length === 0) {
            return;
        }
        
        this.isSyncing = true;
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Starting sync', { items: this.syncQueue.length });
        }
        
        // Emit sync start event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('sync:start', { items: this.syncQueue.length });
        }
        
        const itemsToSync = [...this.syncQueue];
        const results = {
            success: [],
            failed: []
        };
        
        for (const item of itemsToSync) {
            try {
                await this.syncItem(item);
                results.success.push(item);
                this.syncQueue = this.syncQueue.filter(i => i.id !== item.id);
                
                if (typeof Logger !== 'undefined') {
                    Logger.debug('Item synced', { id: item.id });
                }
            } catch (error) {
                item.retries++;
                
                if (item.retries >= 3) {
                    results.failed.push(item);
                    this.syncQueue = this.syncQueue.filter(i => i.id !== item.id);
                    
                    if (typeof Logger !== 'undefined') {
                        Logger.error('Item sync failed after retries', { id: item.id, error });
                    }
                } else {
                    if (typeof Logger !== 'undefined') {
                        Logger.warn('Item sync failed, will retry', { id: item.id, retries: item.retries });
                    }
                }
            }
        }
        
        this.saveQueue();
        this.lastSync = new Date();
        localStorage.setItem('last_sync', this.lastSync.toISOString());
        this.isSyncing = false;
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Sync completed', { 
                success: results.success.length, 
                failed: results.failed.length 
            });
        }
        
        // Emit sync complete event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('sync:complete', results);
        }
        
        // Show notification if there were failures
        if (results.failed.length > 0 && typeof Notifications !== 'undefined') {
            Notifications.warning(`${results.failed.length} items failed to sync`);
        }
    },
    
    /**
     * Sync a single item
     */
    async syncItem(item) {
        // Implement sync logic based on item type
        switch (item.data.type) {
            case 'history':
                // Sync history item
                if (typeof API !== 'undefined') {
                    await API.post('/sync/history', item.data);
                }
                break;
            case 'gallery':
                // Sync gallery item
                if (typeof API !== 'undefined') {
                    await API.post('/sync/gallery', item.data);
                }
                break;
            case 'favorites':
                // Sync favorites
                if (typeof API !== 'undefined') {
                    await API.post('/sync/favorites', item.data);
                }
                break;
            default:
                throw new Error(`Unknown sync type: ${item.data.type}`);
        }
    },
    
    /**
     * Save queue to storage
     */
    saveQueue() {
        try {
            localStorage.setItem('sync_queue', JSON.stringify(this.syncQueue));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save sync queue', error);
            }
        }
    },
    
    /**
     * Load queue from storage
     */
    loadQueue() {
        try {
            const stored = localStorage.getItem('sync_queue');
            if (stored) {
                this.syncQueue = JSON.parse(stored);
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to load sync queue', error);
            }
        }
    },
    
    /**
     * Clear sync queue
     */
    clearQueue() {
        this.syncQueue = [];
        this.saveQueue();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Sync queue cleared');
        }
    },
    
    /**
     * Get sync status
     */
    getStatus() {
        return {
            isSyncing: this.isSyncing,
            queueLength: this.syncQueue.length,
            lastSync: this.lastSync,
            isOnline: typeof OfflineManager !== 'undefined' ? OfflineManager.isOnline : navigator.onLine
        };
    },
    
    /**
     * Force sync
     */
    async forceSync() {
        return await this.sync();
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SyncManager;
}

