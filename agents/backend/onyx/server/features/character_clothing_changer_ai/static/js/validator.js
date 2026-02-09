/**
 * Validator Module
 * ===============
 * Advanced validation utilities
 */

const Validator = {
    /**
     * Validate email
     */
    email(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },

    /**
     * Validate URL
     */
    url(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    },

    /**
     * Validate image file
     */
    imageFile(file) {
        if (!file) return { valid: false, error: 'No file provided' };
        
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
        const maxSize = 10 * 1024 * 1024; // 10MB
        
        if (!validTypes.includes(file.type)) {
            return { 
                valid: false, 
                error: `Invalid file type. Allowed: ${validTypes.join(', ')}` 
            };
        }
        
        if (file.size > maxSize) {
            return { 
                valid: false, 
                error: `File too large. Maximum size: ${(maxSize / 1024 / 1024).toFixed(0)}MB` 
            };
        }
        
        return { valid: true };
    },

    /**
     * Validate text length
     */
    textLength(text, min, max) {
        const length = text.trim().length;
        
        if (min !== undefined && length < min) {
            return { valid: false, error: `Text must be at least ${min} characters` };
        }
        
        if (max !== undefined && length > max) {
            return { valid: false, error: `Text must be at most ${max} characters` };
        }
        
        return { valid: true, length };
    },

    /**
     * Validate number range
     */
    numberRange(number, min, max) {
        const num = parseFloat(number);
        
        if (isNaN(num)) {
            return { valid: false, error: 'Not a valid number' };
        }
        
        if (min !== undefined && num < min) {
            return { valid: false, error: `Number must be at least ${min}` };
        }
        
        if (max !== undefined && num > max) {
            return { valid: false, error: `Number must be at most ${max}` };
        }
        
        return { valid: true, value: num };
    },

    /**
     * Validate required field
     */
    required(value) {
        if (value === null || value === undefined || value === '') {
            return { valid: false, error: 'This field is required' };
        }
        
        if (typeof value === 'string' && value.trim().length === 0) {
            return { valid: false, error: 'This field cannot be empty' };
        }
        
        return { valid: true };
    },

    /**
     * Validate form data
     */
    validateForm(formData, rules) {
        const errors = {};
        let isValid = true;
        
        for (const [field, fieldRules] of Object.entries(rules)) {
            const value = formData[field];
            
            for (const rule of fieldRules) {
                let result;
                
                if (typeof rule === 'function') {
                    result = rule(value, formData);
                } else if (rule.type === 'required') {
                    result = this.required(value);
                } else if (rule.type === 'email') {
                    result = this.email(value) ? { valid: true } : { valid: false, error: 'Invalid email' };
                } else if (rule.type === 'url') {
                    result = this.url(value) ? { valid: true } : { valid: false, error: 'Invalid URL' };
                } else if (rule.type === 'textLength') {
                    result = this.textLength(value, rule.min, rule.max);
                } else if (rule.type === 'numberRange') {
                    result = this.numberRange(value, rule.min, rule.max);
                } else if (rule.type === 'custom') {
                    result = rule.validator(value, formData);
                }
                
                if (result && !result.valid) {
                    errors[field] = result.error;
                    isValid = false;
                    break; // Stop at first error for this field
                }
            }
        }
        
        return { valid: isValid, errors };
    },

    /**
     * Create validation rule
     */
    rule(type, options = {}) {
        return { type, ...options };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Validator;
}

