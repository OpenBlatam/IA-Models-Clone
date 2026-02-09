/**
 * Event Bus Module
 * ================
 * Centralized event system for communication between modules
 */

const EventBus = {
    /**
     * Event listeners storage
     */
    listeners: {},
    
    /**
     * Subscribe to an event
     */
    on(event, callback, context = null) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        
        const listener = {
            callback,
            context,
            id: Date.now() + Math.random()
        };
        
        this.listeners[event].push(listener);
        
        // Return unsubscribe function
        return () => {
            this.off(event, listener.id);
        };
    },
    
    /**
     * Subscribe to an event once
     */
    once(event, callback, context = null) {
        const unsubscribe = this.on(event, (...args) => {
            callback.apply(context, args);
            unsubscribe();
        }, context);
        
        return unsubscribe;
    },
    
    /**
     * Unsubscribe from an event
     */
    off(event, listenerId = null) {
        if (!this.listeners[event]) {
            return;
        }
        
        if (listenerId === null) {
            // Remove all listeners for this event
            delete this.listeners[event];
        } else {
            // Remove specific listener
            this.listeners[event] = this.listeners[event].filter(
                listener => listener.id !== listenerId
            );
            
            if (this.listeners[event].length === 0) {
                delete this.listeners[event];
            }
        }
    },
    
    /**
     * Emit an event
     */
    emit(event, ...args) {
        if (!this.listeners[event]) {
            return;
        }
        
        // Create a copy of listeners to avoid issues if listeners are modified during execution
        const listeners = [...this.listeners[event]];
        
        listeners.forEach(listener => {
            try {
                if (listener.context) {
                    listener.callback.apply(listener.context, args);
                } else {
                    listener.callback(...args);
                }
            } catch (error) {
                console.error(`Error in event listener for "${event}":`, error);
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Event listener error for "${event}":`, error);
                }
            }
        });
    },
    
    /**
     * Get all registered events
     */
    getEvents() {
        return Object.keys(this.listeners);
    },
    
    /**
     * Get listener count for an event
     */
    getListenerCount(event) {
        return this.listeners[event] ? this.listeners[event].length : 0;
    },
    
    /**
     * Clear all listeners
     */
    clear() {
        this.listeners = {};
    },
    
    /**
     * Clear listeners for a specific event
     */
    clearEvent(event) {
        delete this.listeners[event];
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EventBus;
}
