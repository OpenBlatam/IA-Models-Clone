/**
 * Event Store Module
 * =================
 * Event sourcing and event history management
 */

const EventStore = {
    /**
     * Event storage
     */
    events: [],
    
    /**
     * Event handlers
     */
    handlers: new Map(),
    
    /**
     * Snapshot interval
     */
    snapshotInterval: 100, // Create snapshot every 100 events
    
    /**
     * Snapshots
     */
    snapshots: [],
    
    /**
     * Initialize event store
     */
    init() {
        // Load from storage
        this.loadFromStorage();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Event store initialized', { events: this.events.length });
        }
    },
    
    /**
     * Append event
     */
    append(eventType, data, metadata = {}) {
        const event = {
            id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: eventType,
            data,
            metadata: {
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent,
                ...metadata
            },
            version: 1
        };
        
        this.events.push(event);
        
        // Create snapshot if needed
        if (this.events.length % this.snapshotInterval === 0) {
            this.createSnapshot();
        }
        
        // Save to storage
        this.saveToStorage();
        
        // Execute handlers
        this.executeHandlers(event);
        
        // Emit event appended
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('event-store:appended', event);
        }
        
        return event.id;
    },
    
    /**
     * Execute handlers
     */
    executeHandlers(event) {
        const handlers = this.handlers.get(event.type) || [];
        handlers.forEach(handler => {
            try {
                handler(event);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Event handler error for ${event.type}:`, error);
                }
            }
        });
    },
    
    /**
     * Register event handler
     */
    on(eventType, handler) {
        if (!this.handlers.has(eventType)) {
            this.handlers.set(eventType, []);
        }
        
        this.handlers.get(eventType).push(handler);
        
        // Return unsubscribe function
        return () => {
            const handlers = this.handlers.get(eventType);
            if (handlers) {
                const index = handlers.indexOf(handler);
                if (index > -1) {
                    handlers.splice(index, 1);
                }
            }
        };
    },
    
    /**
     * Get events
     */
    getEvents(filter = {}) {
        let events = [...this.events];
        
        // Filter by type
        if (filter.type) {
            events = events.filter(e => e.type === filter.type);
        }
        
        // Filter by date range
        if (filter.from) {
            events = events.filter(e => new Date(e.metadata.timestamp) >= new Date(filter.from));
        }
        if (filter.to) {
            events = events.filter(e => new Date(e.metadata.timestamp) <= new Date(filter.to));
        }
        
        // Limit
        if (filter.limit) {
            events = events.slice(-filter.limit);
        }
        
        return events;
    },
    
    /**
     * Get events by type
     */
    getEventsByType(eventType) {
        return this.events.filter(e => e.type === eventType);
    },
    
    /**
     * Replay events
     */
    replayEvents(events, handlers) {
        events.forEach(event => {
            const eventHandlers = handlers[event.type] || [];
            eventHandlers.forEach(handler => {
                try {
                    handler(event);
                } catch (error) {
                    if (typeof Logger !== 'undefined') {
                        Logger.error(`Replay handler error for ${event.type}:`, error);
                    }
                }
            });
        });
    },
    
    /**
     * Create snapshot
     */
    createSnapshot() {
        const snapshot = {
            id: `snap_${Date.now()}`,
            eventsCount: this.events.length,
            timestamp: new Date().toISOString(),
            state: this.getState()
        };
        
        this.snapshots.push(snapshot);
        
        // Keep only last 10 snapshots
        if (this.snapshots.length > 10) {
            this.snapshots.shift();
        }
        
        // Save snapshot
        try {
            localStorage.setItem('event_store_snapshots', JSON.stringify(this.snapshots));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save snapshot', error);
            }
        }
    },
    
    /**
     * Get state from events
     */
    getState() {
        // This would reconstruct state from events
        // Implementation depends on your state structure
        return {
            totalEvents: this.events.length,
            eventTypes: this.getEventTypes(),
            lastEvent: this.events[this.events.length - 1]
        };
    },
    
    /**
     * Get event types
     */
    getEventTypes() {
        const types = new Set();
        this.events.forEach(event => types.add(event.type));
        return Array.from(types);
    },
    
    /**
     * Clear events
     */
    clear() {
        this.events = [];
        this.snapshots = [];
        
        // Clear storage
        try {
            localStorage.removeItem('event_store');
            localStorage.removeItem('event_store_snapshots');
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to clear event store', error);
            }
        }
    },
    
    /**
     * Save to storage
     */
    saveToStorage() {
        try {
            // Save only last 1000 events
            const eventsToSave = this.events.slice(-1000);
            localStorage.setItem('event_store', JSON.stringify(eventsToSave));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to save event store', error);
            }
        }
    },
    
    /**
     * Load from storage
     */
    loadFromStorage() {
        try {
            const stored = localStorage.getItem('event_store');
            if (stored) {
                this.events = JSON.parse(stored);
            }
            
            const snapshotsStored = localStorage.getItem('event_store_snapshots');
            if (snapshotsStored) {
                this.snapshots = JSON.parse(snapshotsStored);
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Failed to load event store', error);
            }
        }
    },
    
    /**
     * Export events
     */
    exportEvents(filter = {}) {
        const events = this.getEvents(filter);
        const data = {
            events,
            exportedAt: new Date().toISOString(),
            total: events.length
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `events-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    EventStore.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EventStore;
}

