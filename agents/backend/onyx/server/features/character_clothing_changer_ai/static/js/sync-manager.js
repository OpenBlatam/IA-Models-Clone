/**
 * Sync Manager Module
 * ===================
 * Data synchronization across devices/sessions
 */

const SyncManager = {
    /**
     * Sync queue
     */
    syncQueue: [],
    
    /**
     * Last sync timestamp
     */
    lastSync: null,
    
    /**
     * Sync interval (ms)
     */
    syncInterval: 60000, // 1 minute
    
    /**
     * Initialize sync manager
     */
    init() {
        // Load sync queue from storage
        if (typeof Storage !== 'undefined') {
            this.syncQueue = Storage.get('sync_queue', []);
            this.lastSync = Storage.get('last_sync', null);
        }
        
        // Start periodic sync
        this.startPeriodicSync();
        
        // Listen for changes
        if (typeof EventBus !== 'undefined') {
            EventBus.on('state:changed', () => {
                this.queueSync();
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
        setInterval(() => {
            if (typeof OfflineManager !== 'undefined' && OfflineManager.isOnline) {
                this.sync();
            }
        }, this.syncInterval);
    },
    
    /**
     * Queue item for sync
     */
    queueSync(item = null) {
        if (item) {
            this.syncQueue.push({
                ...item,
                timestamp: Date.now(),
                synced: false
            });
        } else {
            // Queue current state
            if (typeof StateManager !== 'undefined') {
                this.syncQueue.push({
                    type: 'state',
                    data: StateManager.getState(),
                    timestamp: Date.now(),
                    synced: false
                });
            }
        }
        
        // Save queue
        if (typeof Storage !== 'undefined') {
            Storage.save('sync_queue', this.syncQueue);
        }
    },
    
    /**
     * Perform sync
     */
    async sync() {
        if (this.syncQueue.length === 0) {
            return;
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Syncing ${this.syncQueue.length} items`);
        }
        
        const itemsToSync = this.syncQueue.filter(item => !item.synced);
        
        for (const item of itemsToSync) {
            try {
                await this.syncItem(item);
                item.synced = true;
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error('Sync item failed:', error);
                }
            }
        }
        
        // Remove synced items
        this.syncQueue = this.syncQueue.filter(item => !item.synced);
        
        // Update last sync
        this.lastSync = Date.now();
        
        // Save state
        if (typeof Storage !== 'undefined') {
            Storage.save('sync_queue', this.syncQueue);
            Storage.save('last_sync', this.lastSync);
        }
        
        // Emit sync event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('sync:completed', {
                synced: itemsToSync.length,
                timestamp: this.lastSync
            });
        }
    },
    
    /**
     * Sync individual item
     */
    async syncItem(item) {
        // This would sync with a server or other storage
        // For now, just mark as synced
        return Promise.resolve();
    },
    
    /**
     * Get sync status
     */
    getStatus() {
        return {
            queueLength: this.syncQueue.length,
            lastSync: this.lastSync,
            pendingItems: this.syncQueue.filter(item => !item.synced).length
        };
    },
    
    /**
     * Clear sync queue
     */
    clearQueue() {
        this.syncQueue = [];
        
        if (typeof Storage !== 'undefined') {
            Storage.save('sync_queue', []);
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Sync queue cleared');
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SyncManager;
}

