/**
 * Type Guards Module
 * ==================
 * Utility functions for type checking and validation
 */

const TypeGuards = {
    /**
     * Check if value is a string
     */
    isString(value) {
        return typeof value === 'string';
    },
    
    /**
     * Check if value is a number
     */
    isNumber(value) {
        return typeof value === 'number' && !isNaN(value);
    },
    
    /**
     * Check if value is a boolean
     */
    isBoolean(value) {
        return typeof value === 'boolean';
    },
    
    /**
     * Check if value is an object (not null, not array)
     */
    isObject(value) {
        return value !== null && typeof value === 'object' && !Array.isArray(value);
    },
    
    /**
     * Check if value is an array
     */
    isArray(value) {
        return Array.isArray(value);
    },
    
    /**
     * Check if value is a function
     */
    isFunction(value) {
        return typeof value === 'function';
    },
    
    /**
     * Check if value is null or undefined
     */
    isNullOrUndefined(value) {
        return value === null || value === undefined;
    },
    
    /**
     * Check if value is a valid Date
     */
    isDate(value) {
        return value instanceof Date && !isNaN(value.getTime());
    },
    
    /**
     * Check if value is a File
     */
    isFile(value) {
        return value instanceof File;
    },
    
    /**
     * Check if value is a Blob
     */
    isBlob(value) {
        return value instanceof Blob;
    },
    
    /**
     * Check if value is a valid URL
     */
    isURL(value) {
        if (!this.isString(value)) return false;
        try {
            new URL(value);
            return true;
        } catch {
            return false;
        }
    },
    
    /**
     * Check if value is a valid email
     */
    isEmail(value) {
        if (!this.isString(value)) return false;
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(value);
    },
    
    /**
     * Check if value is a valid base64 string
     */
    isBase64(value) {
        if (!this.isString(value)) return false;
        const base64Regex = /^data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+)?;base64,([A-Za-z0-9+/=]+)$/;
        return base64Regex.test(value);
    },
    
    /**
     * Check if value is a valid image data URL
     */
    isImageDataURL(value) {
        if (!this.isString(value)) return false;
        return value.startsWith('data:image/') && this.isBase64(value);
    },
    
    /**
     * Check if value is a valid API response
     */
    isAPIResponse(value) {
        if (!this.isObject(value)) return false;
        return 'success' in value && this.isBoolean(value.success);
    },
    
    /**
     * Check if value is a valid ClothingChangeResult
     */
    isClothingChangeResult(value) {
        if (!this.isObject(value)) return false;
        return 'clothing_description' in value && 
               'changed' in value && 
               this.isString(value.clothing_description) &&
               this.isBoolean(value.changed);
    },
    
    /**
     * Check if value is a valid GalleryItem
     */
    isGalleryItem(value) {
        if (!this.isObject(value)) return false;
        return 'id' in value && 
               'timestamp' in value && 
               this.isNumber(value.id) &&
               this.isString(value.timestamp);
    },
    
    /**
     * Check if value is a valid HistoryItem
     */
    isHistoryItem(value) {
        if (!this.isObject(value)) return false;
        return 'id' in value && 
               'timestamp' in value && 
               this.isNumber(value.id) &&
               this.isString(value.timestamp);
    },
    
    /**
     * Check if value is a valid FormData
     */
    isFormData(value) {
        return value instanceof FormData;
    },
    
    /**
     * Check if value is empty (null, undefined, empty string, empty array, empty object)
     */
    isEmpty(value) {
        if (this.isNullOrUndefined(value)) return true;
        if (this.isString(value)) return value.trim().length === 0;
        if (this.isArray(value)) return value.length === 0;
        if (this.isObject(value)) return Object.keys(value).length === 0;
        return false;
    },
    
    /**
     * Check if value is a positive number
     */
    isPositiveNumber(value) {
        return this.isNumber(value) && value > 0;
    },
    
    /**
     * Check if value is a non-negative number
     */
    isNonNegativeNumber(value) {
        return this.isNumber(value) && value >= 0;
    },
    
    /**
     * Check if value is within a range
     */
    isInRange(value, min, max) {
        if (!this.isNumber(value)) return false;
        return value >= min && value <= max;
    },
    
    /**
     * Check if value has a minimum length
     */
    hasMinLength(value, minLength) {
        if (this.isString(value)) return value.length >= minLength;
        if (this.isArray(value)) return value.length >= minLength;
        return false;
    },
    
    /**
     * Check if value has a maximum length
     */
    hasMaxLength(value, maxLength) {
        if (this.isString(value)) return value.length <= maxLength;
        if (this.isArray(value)) return value.length <= maxLength;
        return false;
    },
    
    /**
     * Check if value matches a pattern
     */
    matchesPattern(value, pattern) {
        if (!this.isString(value)) return false;
        if (this.isString(pattern)) {
            const regex = new RegExp(pattern);
            return regex.test(value);
        }
        if (pattern instanceof RegExp) {
            return pattern.test(value);
        }
        return false;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TypeGuards;
}

