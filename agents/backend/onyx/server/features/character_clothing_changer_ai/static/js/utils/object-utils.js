/**
 * Object Utils Module
 * ===================
 * Utilities for object manipulation and operations
 */

const ObjectUtils = {
    /**
     * Deep clone object
     */
    deepClone(obj) {
        if (obj === null || typeof obj !== 'object') {
            return obj;
        }
        
        if (obj instanceof Date) {
            return new Date(obj.getTime());
        }
        
        if (obj instanceof Array) {
            return obj.map(item => this.deepClone(item));
        }
        
        if (typeof obj === 'object') {
            const cloned = {};
            Object.keys(obj).forEach(key => {
                cloned[key] = this.deepClone(obj[key]);
            });
            return cloned;
        }
        
        return obj;
    },
    
    /**
     * Deep merge objects
     */
    deepMerge(target, ...sources) {
        if (!sources.length) return target;
        const source = sources.shift();
        
        if (this.isObject(target) && this.isObject(source)) {
            for (const key in source) {
                if (this.isObject(source[key])) {
                    if (!target[key]) Object.assign(target, { [key]: {} });
                    this.deepMerge(target[key], source[key]);
                } else {
                    Object.assign(target, { [key]: source[key] });
                }
            }
        }
        
        return this.deepMerge(target, ...sources);
    },
    
    /**
     * Check if value is object
     */
    isObject(value) {
        return value !== null && typeof value === 'object' && !Array.isArray(value);
    },
    
    /**
     * Pick properties from object
     */
    pick(obj, keys) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const picked = {};
        keys.forEach(key => {
            if (key in obj) {
                picked[key] = obj[key];
            }
        });
        return picked;
    },
    
    /**
     * Omit properties from object
     */
    omit(obj, keys) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const omitted = {};
        Object.keys(obj).forEach(key => {
            if (!keys.includes(key)) {
                omitted[key] = obj[key];
            }
        });
        return omitted;
    },
    
    /**
     * Get nested property
     */
    get(obj, path, defaultValue = undefined) {
        if (!this.isObject(obj)) {
            return defaultValue;
        }
        
        const keys = path.split('.');
        let result = obj;
        
        for (const key of keys) {
            if (result === null || result === undefined) {
                return defaultValue;
            }
            result = result[key];
        }
        
        return result !== undefined ? result : defaultValue;
    },
    
    /**
     * Set nested property
     */
    set(obj, path, value) {
        if (!this.isObject(obj)) {
            return obj;
        }
        
        const keys = path.split('.');
        const lastKey = keys.pop();
        let current = obj;
        
        for (const key of keys) {
            if (!this.isObject(current[key])) {
                current[key] = {};
            }
            current = current[key];
        }
        
        current[lastKey] = value;
        return obj;
    },
    
    /**
     * Check if object has nested property
     */
    has(obj, path) {
        if (!this.isObject(obj)) {
            return false;
        }
        
        const keys = path.split('.');
        let current = obj;
        
        for (const key of keys) {
            if (current === null || current === undefined || !(key in current)) {
                return false;
            }
            current = current[key];
        }
        
        return true;
    },
    
    /**
     * Flatten object
     */
    flatten(obj, prefix = '', result = {}) {
        if (!this.isObject(obj)) {
            return result;
        }
        
        Object.keys(obj).forEach(key => {
            const newKey = prefix ? `${prefix}.${key}` : key;
            const value = obj[key];
            
            if (this.isObject(value) && !Array.isArray(value)) {
                this.flatten(value, newKey, result);
            } else {
                result[newKey] = value;
            }
        });
        
        return result;
    },
    
    /**
     * Unflatten object
     */
    unflatten(obj) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const result = {};
        
        Object.keys(obj).forEach(key => {
            this.set(result, key, obj[key]);
        });
        
        return result;
    },
    
    /**
     * Map object values
     */
    mapValues(obj, fn) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const result = {};
        Object.keys(obj).forEach(key => {
            result[key] = fn(obj[key], key, obj);
        });
        return result;
    },
    
    /**
     * Map object keys
     */
    mapKeys(obj, fn) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const result = {};
        Object.keys(obj).forEach(key => {
            const newKey = fn(key, obj[key], obj);
            result[newKey] = obj[key];
        });
        return result;
    },
    
    /**
     * Invert object
     */
    invert(obj) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const result = {};
        Object.keys(obj).forEach(key => {
            result[obj[key]] = key;
        });
        return result;
    },
    
    /**
     * Get object size
     */
    size(obj) {
        if (!this.isObject(obj)) {
            return 0;
        }
        
        return Object.keys(obj).length;
    },
    
    /**
     * Check if object is empty
     */
    isEmpty(obj) {
        if (!this.isObject(obj)) {
            return true;
        }
        
        return Object.keys(obj).length === 0;
    },
    
    /**
     * Omit null/undefined values
     */
    compact(obj) {
        if (!this.isObject(obj)) {
            return {};
        }
        
        const result = {};
        Object.keys(obj).forEach(key => {
            if (obj[key] !== null && obj[key] !== undefined) {
                result[key] = obj[key];
            }
        });
        return result;
    },
    
    /**
     * Defaults object
     */
    defaults(obj, ...defaults) {
        const result = this.deepClone(obj);
        
        defaults.reverse().forEach(defaultObj => {
            Object.keys(defaultObj).forEach(key => {
                if (!(key in result) || result[key] === undefined) {
                    result[key] = defaultObj[key];
                }
            });
        });
        
        return result;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ObjectUtils;
}

