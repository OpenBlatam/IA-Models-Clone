/**
 * Command Pattern Module
 * ======================
 * Command pattern implementation for undo/redo functionality
 */

const CommandPattern = {
    /**
     * Command history
     */
    history: [],
    
    /**
     * Current position in history
     */
    currentPosition: -1,
    
    /**
     * Max history size
     */
    maxHistory: 50,
    
    /**
     * Initialize command pattern
     */
    init(options = {}) {
        this.maxHistory = options.maxHistory || this.maxHistory;
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Command pattern initialized');
        }
    },
    
    /**
     * Execute command
     */
    execute(command) {
        // Remove any commands after current position (for redo)
        if (this.currentPosition < this.history.length - 1) {
            this.history = this.history.slice(0, this.currentPosition + 1);
        }
        
        // Execute command
        const result = command.execute();
        
        // Add to history
        this.history.push(command);
        this.currentPosition = this.history.length - 1;
        
        // Limit history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
            this.currentPosition--;
        }
        
        // Emit command executed event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('command:executed', { command, result });
        }
        
        return result;
    },
    
    /**
     * Undo last command
     */
    undo() {
        if (this.currentPosition < 0) {
            return false;
        }
        
        const command = this.history[this.currentPosition];
        const result = command.undo();
        
        this.currentPosition--;
        
        // Emit command undone event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('command:undone', { command, result });
        }
        
        return result;
    },
    
    /**
     * Redo last undone command
     */
    redo() {
        if (this.currentPosition >= this.history.length - 1) {
            return false;
        }
        
        this.currentPosition++;
        const command = this.history[this.currentPosition];
        const result = command.execute();
        
        // Emit command redone event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('command:redone', { command, result });
        }
        
        return result;
    },
    
    /**
     * Check if can undo
     */
    canUndo() {
        return this.currentPosition >= 0;
    },
    
    /**
     * Check if can redo
     */
    canRedo() {
        return this.currentPosition < this.history.length - 1;
    },
    
    /**
     * Clear history
     */
    clear() {
        this.history = [];
        this.currentPosition = -1;
        
        // Emit history cleared event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('command:history_cleared');
        }
    },
    
    /**
     * Get history
     */
    getHistory() {
        return [...this.history];
    },
    
    /**
     * Get current position
     */
    getCurrentPosition() {
        return this.currentPosition;
    },
    
    /**
     * Create command
     */
    createCommand(name, executeFn, undoFn, data = {}) {
        return {
            name,
            data,
            execute: executeFn,
            undo: undoFn,
            timestamp: Date.now()
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CommandPattern;
}

