/**
 * Config Manager Module
 * ====================
 * Advanced configuration management with validation, persistence, and change tracking
 */

const ConfigManager = {
    /**
     * Configuration storage
     */
    config: {},
    
    /**
     * Default configuration
     */
    defaults: {},
    
    /**
     * Configuration schema for validation
     */
    schema: {},
    
    /**
     * Configuration change listeners
     */
    listeners: {},
    
    /**
     * Initialize configuration
     */
    init(defaultConfig = {}, schema = {}) {
        this.defaults = { ...defaultConfig };
        this.schema = { ...schema };
        
        // Load from storage
        this.load();
        
        // Merge with defaults
        this.config = this.mergeWithDefaults(this.config, this.defaults);
        
        // Validate configuration
        this.validate();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Config manager initialized');
        }
    },
    
    /**
     * Get configuration value
     */
    get(key, defaultValue = undefined) {
        if (key === undefined) {
            return { ...this.config };
        }
        
        const keys = key.split('.');
        let value = this.config;
        
        for (const k of keys) {
            if (value === null || value === undefined) {
                return defaultValue;
            }
            value = value[k];
        }
        
        return value !== undefined ? value : defaultValue;
    },
    
    /**
     * Set configuration value
     */
    set(key, value) {
        const keys = key.split('.');
        const lastKey = keys.pop();
        let current = this.config;
        
        // Navigate/create nested structure
        for (const k of keys) {
            if (!this.isObject(current[k])) {
                current[k] = {};
            }
            current = current[k];
        }
        
        const oldValue = current[lastKey];
        current[lastKey] = value;
        
        // Validate
        if (this.schema[key]) {
            const validation = this.validateValue(key, value);
            if (!validation.valid) {
                current[lastKey] = oldValue; // Revert
                throw new Error(`Invalid config value for ${key}: ${validation.error}`);
            }
        }
        
        // Save to storage
        this.save();
        
        // Emit change event
        this.emitChange(key, value, oldValue);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Config changed: ${key}`, { oldValue, newValue: value });
        }
        
        return this;
    },
    
    /**
     * Update multiple configuration values
     */
    update(updates) {
        const changes = {};
        
        Object.keys(updates).forEach(key => {
            const oldValue = this.get(key);
            this.set(key, updates[key]);
            changes[key] = { oldValue, newValue: updates[key] };
        });
        
        // Emit batch change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('config:batch_changed', changes);
        }
        
        return this;
    },
    
    /**
     * Reset configuration to defaults
     */
    reset(key = null) {
        if (key === null) {
            this.config = { ...this.defaults };
        } else {
            this.set(key, this.defaults[key]);
        }
        
        this.save();
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Config reset${key ? `: ${key}` : ''}`);
        }
        
        return this;
    },
    
    /**
     * Validate configuration
     */
    validate() {
        const errors = [];
        
        Object.keys(this.schema).forEach(key => {
            const value = this.get(key);
            if (value !== undefined) {
                const validation = this.validateValue(key, value);
                if (!validation.valid) {
                    errors.push(`${key}: ${validation.error}`);
                }
            }
        });
        
        if (errors.length > 0) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Config validation errors:', errors);
            }
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    },
    
    /**
     * Validate a single value
     */
    validateValue(key, value) {
        const rule = this.schema[key];
        if (!rule) {
            return { valid: true };
        }
        
        // Type validation
        if (rule.type && typeof value !== rule.type) {
            return { valid: false, error: `Expected type ${rule.type}, got ${typeof value}` };
        }
        
        // Required validation
        if (rule.required && (value === undefined || value === null)) {
            return { valid: false, error: 'Required field' };
        }
        
        // Min/Max validation for numbers
        if (rule.type === 'number') {
            if (rule.min !== undefined && value < rule.min) {
                return { valid: false, error: `Value must be >= ${rule.min}` };
            }
            if (rule.max !== undefined && value > rule.max) {
                return { valid: false, error: `Value must be <= ${rule.max}` };
            }
        }
        
        // Min/Max length validation for strings/arrays
        if (rule.type === 'string' || Array.isArray(value)) {
            if (rule.minLength !== undefined && value.length < rule.minLength) {
                return { valid: false, error: `Length must be >= ${rule.minLength}` };
            }
            if (rule.maxLength !== undefined && value.length > rule.maxLength) {
                return { valid: false, error: `Length must be <= ${rule.maxLength}` };
            }
        }
        
        // Enum validation
        if (rule.enum && !rule.enum.includes(value)) {
            return { valid: false, error: `Value must be one of: ${rule.enum.join(', ')}` };
        }
        
        // Custom validator
        if (rule.validator && typeof rule.validator === 'function') {
            const result = rule.validator(value);
            if (result !== true) {
                return { valid: false, error: result || 'Validation failed' };
            }
        }
        
        return { valid: true };
    },
    
    /**
     * Subscribe to configuration changes
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
     * Unsubscribe from configuration changes
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
     * Emit configuration change
     */
    emitChange(key, newValue, oldValue) {
        if (!this.listeners[key]) {
            return;
        }
        
        this.listeners[key].forEach(callback => {
            try {
                callback(newValue, oldValue, key);
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Error in config listener for ${key}:`, error);
                }
            }
        });
        
        // Emit global change event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('config:changed', { key, newValue, oldValue });
        }
    },
    
    /**
     * Load configuration from storage
     */
    load() {
        try {
            const stored = localStorage.getItem('app_config');
            if (stored) {
                this.config = JSON.parse(stored);
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Error loading config from storage:', error);
            }
        }
    },
    
    /**
     * Save configuration to storage
     */
    save() {
        try {
            localStorage.setItem('app_config', JSON.stringify(this.config));
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.warn('Error saving config to storage:', error);
            }
        }
    },
    
    /**
     * Merge with defaults
     */
    mergeWithDefaults(config, defaults) {
        const merged = { ...defaults };
        
        Object.keys(config).forEach(key => {
            if (this.isObject(config[key]) && this.isObject(defaults[key])) {
                merged[key] = this.mergeWithDefaults(config[key], defaults[key]);
            } else {
                merged[key] = config[key];
            }
        });
        
        return merged;
    },
    
    /**
     * Check if value is object
     */
    isObject(value) {
        return value !== null && typeof value === 'object' && !Array.isArray(value);
    },
    
    /**
     * Export configuration
     */
    export() {
        return JSON.stringify(this.config, null, 2);
    },
    
    /**
     * Import configuration
     */
    import(configJson) {
        try {
            const imported = JSON.parse(configJson);
            this.update(imported);
            
            if (typeof Logger !== 'undefined') {
                Logger.info('Config imported successfully');
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Error importing config:', error);
            }
            throw error;
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConfigManager;
}

