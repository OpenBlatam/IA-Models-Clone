/**
 * Logger Module
 * ============
 * Centralized logging system with different log levels
 */

const Logger = {
    /**
     * Log levels
     */
    LEVELS: {
        DEBUG: 0,
        INFO: 1,
        WARN: 2,
        ERROR: 3,
        NONE: 4
    },
    
    /**
     * Current log level (default: INFO in production, DEBUG in development)
     */
    currentLevel: window.location.hostname === 'localhost' 
        ? this?.LEVELS?.DEBUG || 0 
        : this?.LEVELS?.INFO || 1,
    
    /**
     * Enable/disable console logging
     */
    enableConsole: true,
    
    /**
     * Log history (for debugging)
     */
    history: [],
    maxHistorySize: 100,
    
    /**
     * Log a message
     */
    log(level, message, ...args) {
        if (level < this.currentLevel) {
            return;
        }
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level: Object.keys(this.LEVELS).find(key => this.LEVELS[key] === level),
            message,
            args: args.length > 0 ? args : undefined
        };
        
        // Add to history
        this.history.push(logEntry);
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }
        
        // Console logging
        if (this.enableConsole) {
            const levelName = logEntry.level || 'LOG';
            const prefix = `[${timestamp}] [${levelName}]`;
            
            switch (level) {
                case this.LEVELS.DEBUG:
                    console.debug(prefix, message, ...args);
                    break;
                case this.LEVELS.INFO:
                    console.info(prefix, message, ...args);
                    break;
                case this.LEVELS.WARN:
                    console.warn(prefix, message, ...args);
                    break;
                case this.LEVELS.ERROR:
                    console.error(prefix, message, ...args);
                    break;
                default:
                    console.log(prefix, message, ...args);
            }
        }
        
        // Emit log event if EventBus is available
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('log', logEntry);
        }
    },
    
    /**
     * Debug log
     */
    debug(message, ...args) {
        this.log(this.LEVELS.DEBUG, message, ...args);
    },
    
    /**
     * Info log
     */
    info(message, ...args) {
        this.log(this.LEVELS.INFO, message, ...args);
    },
    
    /**
     * Warning log
     */
    warn(message, ...args) {
        this.log(this.LEVELS.WARN, message, ...args);
    },
    
    /**
     * Error log
     */
    error(message, ...args) {
        this.log(this.LEVELS.ERROR, message, ...args);
    },
    
    /**
     * Set log level
     */
    setLevel(level) {
        if (typeof level === 'string') {
            level = this.LEVELS[level.toUpperCase()];
        }
        if (level !== undefined) {
            this.currentLevel = level;
        }
    },
    
    /**
     * Get log history
     */
    getHistory(filterLevel = null) {
        if (filterLevel === null) {
            return this.history;
        }
        const level = typeof filterLevel === 'string' 
            ? this.LEVELS[filterLevel.toUpperCase()] 
            : filterLevel;
        return this.history.filter(entry => {
            const entryLevel = this.LEVELS[entry.level];
            return entryLevel >= level;
        });
    },
    
    /**
     * Clear log history
     */
    clearHistory() {
        this.history = [];
    },
    
    /**
     * Export log history
     */
    exportHistory() {
        return JSON.stringify(this.history, null, 2);
    }
};

// Initialize log level
Logger.currentLevel = window.location.hostname === 'localhost' 
    ? Logger.LEVELS.DEBUG 
    : Logger.LEVELS.INFO;

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Logger;
}

