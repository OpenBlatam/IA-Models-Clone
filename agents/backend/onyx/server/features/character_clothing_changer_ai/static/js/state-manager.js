/**
 * State Manager Module
 * ====================
 * Centralized state management for the application
 */

const StateManager = {
    /**
     * Application state
     */
    state: {
        currentResult: null,
        currentImage: null,
        currentTab: 'result',
        theme: 'default',
        isProcessing: false,
        serverStatus: 'unknown',
        modelInfo: null
    },
    
    /**
     * State change listeners
     */
    listeners: {},
    
    /**
     * Get state value
     */
    get(key) {
        return this.state[key];
    },
    
    /**
     * Set state value
     */
    set(key, value) {
        const oldValue = this.state[key];
        this.state[key] = value;
        
        // Emit state change event
        this.emitChange(key, value, oldValue);
        
        // Emit specific key change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit(`state:${key}`, value, oldValue);
        }
        
        // Log state change in debug mode
        if (typeof Logger !== 'undefined') {
            Logger.debug(`State changed: ${key}`, { oldValue, newValue: value });
        }
    },
    
    /**
     * Update multiple state values at once
     */
    update(updates) {
        const changes = {};
        
        Object.keys(updates).forEach(key => {
            const oldValue = this.state[key];
            this.state[key] = updates[key];
            changes[key] = { oldValue, newValue: updates[key] };
        });
        
        // Emit batch change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('state:batch', changes);
        }
        
        // Emit individual change events
        Object.keys(changes).forEach(key => {
            this.emitChange(key, changes[key].newValue, changes[key].oldValue);
            if (typeof EventBus !== 'undefined') {
                EventBus.emit(`state:${key}`, changes[key].newValue, changes[key].oldValue);
            }
        });
    },
    
    /**
     * Subscribe to state changes
     */
    subscribe(key, callback) {
        if (!this.listeners[key]) {
            this.listeners[key] = [];
        }
        
        this.listeners[key].push(callback);
        
        // Return unsubscribe function
        return () => {
            this.unsubscribe(key, callback);
        };
    },
    
    /**
     * Unsubscribe from state changes
     */
    unsubscribe(key, callback) {
        if (!this.listeners[key]) {
            return;
        }
        
        this.listeners[key] = this.listeners[key].filter(cb => cb !== callback);
        
        if (this.listeners[key].length === 0) {
            delete this.listeners[key];
        }
    },
    
    /**
     * Emit state change to listeners
     */
    emitChange(key, newValue, oldValue) {
        if (!this.listeners[key]) {
            return;
        }
        
        this.listeners[key].forEach(callback => {
            try {
                callback(newValue, oldValue, key);
            } catch (error) {
                console.error(`Error in state listener for "${key}":`, error);
                if (typeof Logger !== 'undefined') {
                    Logger.error(`State listener error for "${key}":`, error);
                }
            }
        });
    },
    
    /**
     * Reset state to initial values
     */
    reset() {
        const initialState = {
            currentResult: null,
            currentImage: null,
            currentTab: 'result',
            theme: 'default',
            isProcessing: false,
            serverStatus: 'unknown',
            modelInfo: null
        };
        
        this.update(initialState);
    },
    
    /**
     * Get entire state (for debugging)
     */
    getState() {
        return { ...this.state };
    },
    
    /**
     * Export state (for persistence)
     */
    export() {
        return JSON.stringify(this.state, null, 2);
    },
    
    /**
     * Import state (for restoration)
     */
    import(stateJson) {
        try {
            const importedState = JSON.parse(stateJson);
            this.update(importedState);
        } catch (error) {
            console.error('Error importing state:', error);
            if (typeof Logger !== 'undefined') {
                Logger.error('State import error:', error);
            }
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StateManager;
}
