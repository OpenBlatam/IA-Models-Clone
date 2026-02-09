/**
 * State Machine Module
 * ====================
 * Finite state machine for managing application states
 */

const StateMachine = {
    /**
     * State machines
     */
    machines: new Map(),
    
    /**
     * Create state machine
     */
    create(name, initialState, states, transitions) {
        const machine = {
            name,
            currentState: initialState,
            initialState,
            states: new Map(Object.entries(states)),
            transitions: new Map(Object.entries(transitions)),
            history: [],
            listeners: new Set()
        };
        
        this.machines.set(name, machine);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`State machine created: ${name}`, { initialState });
        }
        
        return machine;
    },
    
    /**
     * Get state machine
     */
    get(name) {
        return this.machines.get(name);
    },
    
    /**
     * Transition state
     */
    transition(machineName, event, data = {}) {
        const machine = this.machines.get(machineName);
        if (!machine) {
            throw new Error(`State machine not found: ${machineName}`);
        }
        
        const currentState = machine.currentState;
        const transitionKey = `${currentState}:${event}`;
        const transition = machine.transitions.get(transitionKey);
        
        if (!transition) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Invalid transition: ${transitionKey}`);
            }
            return false;
        }
        
        // Check guard
        if (transition.guard && !transition.guard(currentState, event, data)) {
            if (typeof Logger !== 'undefined') {
                Logger.warn(`Transition guard failed: ${transitionKey}`);
            }
            return false;
        }
        
        // Get target state
        const targetState = typeof transition.target === 'function' 
            ? transition.target(currentState, event, data)
            : transition.target;
        
        // Validate target state
        if (!machine.states.has(targetState)) {
            throw new Error(`Invalid target state: ${targetState}`);
        }
        
        // Execute onExit for current state
        const currentStateConfig = machine.states.get(currentState);
        if (currentStateConfig && currentStateConfig.onExit) {
            currentStateConfig.onExit(currentState, targetState, data);
        }
        
        // Execute transition action
        if (transition.action) {
            transition.action(currentState, targetState, data);
        }
        
        // Update state
        const previousState = machine.currentState;
        machine.currentState = targetState;
        
        // Add to history
        machine.history.push({
            from: previousState,
            to: targetState,
            event,
            timestamp: Date.now(),
            data
        });
        
        // Keep only last 100 entries
        if (machine.history.length > 100) {
            machine.history.shift();
        }
        
        // Execute onEnter for target state
        const targetStateConfig = machine.states.get(targetState);
        if (targetStateConfig && targetStateConfig.onEnter) {
            targetStateConfig.onEnter(previousState, targetState, data);
        }
        
        // Notify listeners
        machine.listeners.forEach(listener => {
            try {
                listener({
                    from: previousState,
                    to: targetState,
                    event,
                    data
                });
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error('State machine listener error', error);
                }
            }
        });
        
        // Emit state change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('state-machine:changed', {
                machine: machineName,
                from: previousState,
                to: targetState,
                event,
                data
            });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`State transition: ${machineName} ${previousState} -> ${targetState} (${event})`);
        }
        
        return true;
    },
    
    /**
     * Get current state
     */
    getState(machineName) {
        const machine = this.machines.get(machineName);
        return machine ? machine.currentState : null;
    },
    
    /**
     * Check if can transition
     */
    canTransition(machineName, event) {
        const machine = this.machines.get(machineName);
        if (!machine) {
            return false;
        }
        
        const currentState = machine.currentState;
        const transitionKey = `${currentState}:${event}`;
        const transition = machine.transitions.get(transitionKey);
        
        if (!transition) {
            return false;
        }
        
        // Check guard if exists
        if (transition.guard) {
            return transition.guard(currentState, event, {});
        }
        
        return true;
    },
    
    /**
     * Subscribe to state changes
     */
    subscribe(machineName, listener) {
        const machine = this.machines.get(machineName);
        if (!machine) {
            throw new Error(`State machine not found: ${machineName}`);
        }
        
        machine.listeners.add(listener);
        
        // Return unsubscribe function
        return () => {
            machine.listeners.delete(listener);
        };
    },
    
    /**
     * Reset state machine
     */
    reset(machineName) {
        const machine = this.machines.get(machineName);
        if (!machine) {
            return false;
        }
        
        machine.currentState = machine.initialState;
        machine.history = [];
        
        // Emit reset event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('state-machine:reset', { machine: machineName });
        }
        
        return true;
    },
    
    /**
     * Get state history
     */
    getHistory(machineName) {
        const machine = this.machines.get(machineName);
        return machine ? [...machine.history] : [];
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StateMachine;
}

