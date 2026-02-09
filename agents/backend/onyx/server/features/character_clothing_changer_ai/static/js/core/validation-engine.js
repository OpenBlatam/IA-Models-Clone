/**
 * Validation Engine Module
 * ========================
 * Advanced validation system with rules and custom validators
 */

const ValidationEngine = {
    /**
     * Validation rules
     */
    rules: new Map(),
    
    /**
     * Custom validators
     */
    validators: new Map(),
    
    /**
     * Validation errors
     */
    errors: new Map(),
    
    /**
     * Initialize validation engine
     */
    init() {
        // Register built-in validators
        this.registerBuiltInValidators();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Validation engine initialized');
        }
    },
    
    /**
     * Register built-in validators
     */
    registerBuiltInValidators() {
        // Required validator
        this.registerValidator('required', (value) => {
            if (value === null || value === undefined || value === '') {
                return { valid: false, message: 'Este campo es requerido' };
            }
            return { valid: true };
        });
        
        // Email validator
        this.registerValidator('email', (value) => {
            if (!value) return { valid: true }; // Skip if empty (use required for that)
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return {
                valid: emailRegex.test(value),
                message: 'Email inválido'
            };
        });
        
        // Min length validator
        this.registerValidator('minLength', (value, min) => {
            if (!value) return { valid: true };
            return {
                valid: value.length >= min,
                message: `Debe tener al menos ${min} caracteres`
            };
        });
        
        // Max length validator
        this.registerValidator('maxLength', (value, max) => {
            if (!value) return { valid: true };
            return {
                valid: value.length <= max,
                message: `No puede exceder ${max} caracteres`
            };
        });
        
        // Pattern validator
        this.registerValidator('pattern', (value, pattern) => {
            if (!value) return { valid: true };
            const regex = new RegExp(pattern);
            return {
                valid: regex.test(value),
                message: 'Formato inválido'
            };
        });
        
        // Number range validator
        this.registerValidator('range', (value, min, max) => {
            if (value === null || value === undefined || value === '') {
                return { valid: true };
            }
            const num = Number(value);
            return {
                valid: !isNaN(num) && num >= min && num <= max,
                message: `Debe estar entre ${min} y ${max}`
            };
        });
        
        // File type validator
        this.registerValidator('fileType', (file, allowedTypes) => {
            if (!file) return { valid: true };
            const types = Array.isArray(allowedTypes) ? allowedTypes : [allowedTypes];
            return {
                valid: types.includes(file.type),
                message: `Tipo de archivo no permitido. Permitidos: ${types.join(', ')}`
            };
        });
        
        // File size validator
        this.registerValidator('fileSize', (file, maxSize) => {
            if (!file) return { valid: true };
            return {
                valid: file.size <= maxSize,
                message: `El archivo es demasiado grande. Máximo: ${this.formatBytes(maxSize)}`
            };
        });
    },
    
    /**
     * Register validator
     */
    registerValidator(name, validator) {
        this.validators.set(name, validator);
        
        if (typeof Logger !== 'undefined') {
            Logger.debug(`Validator registered: ${name}`);
        }
    },
    
    /**
     * Register validation rule
     */
    registerRule(field, rule) {
        if (!this.rules.has(field)) {
            this.rules.set(field, []);
        }
        
        this.rules.get(field).push(rule);
    },
    
    /**
     * Validate field
     */
    validateField(field, value) {
        const fieldRules = this.rules.get(field) || [];
        const errors = [];
        
        for (const rule of fieldRules) {
            const validator = this.validators.get(rule.validator);
            if (!validator) {
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Validator not found: ${rule.validator}`);
                }
                continue;
            }
            
            const result = validator(value, ...(rule.params || []));
            if (!result.valid) {
                errors.push({
                    field,
                    validator: rule.validator,
                    message: result.message || rule.message || 'Validación fallida'
                });
            }
        }
        
        if (errors.length > 0) {
            this.errors.set(field, errors);
        } else {
            this.errors.delete(field);
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    },
    
    /**
     * Validate form
     */
    validateForm(formData) {
        this.errors.clear();
        const results = {};
        let allValid = true;
        
        // Validate all registered fields
        this.rules.forEach((rules, field) => {
            const value = this.getFieldValue(formData, field);
            const result = this.validateField(field, value);
            results[field] = result;
            
            if (!result.valid) {
                allValid = false;
            }
        });
        
        return {
            valid: allValid,
            results,
            errors: Array.from(this.errors.entries()).map(([field, errors]) => ({
                field,
                errors
            }))
        };
    },
    
    /**
     * Get field value from form data
     */
    getFieldValue(formData, field) {
        if (formData instanceof FormData) {
            return formData.get(field);
        }
        if (typeof formData === 'object') {
            return formData[field];
        }
        return null;
    },
    
    /**
     * Clear errors
     */
    clearErrors(field = null) {
        if (field) {
            this.errors.delete(field);
        } else {
            this.errors.clear();
        }
    },
    
    /**
     * Get errors
     */
    getErrors(field = null) {
        if (field) {
            return this.errors.get(field) || [];
        }
        return Array.from(this.errors.entries()).map(([field, errors]) => ({
            field,
            errors
        }));
    },
    
    /**
     * Format bytes helper
     */
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    /**
     * Setup form validation
     */
    setupForm(formId, rules) {
        const form = document.getElementById(formId);
        if (!form) {
            throw new Error(`Form not found: ${formId}`);
        }
        
        // Register rules
        Object.keys(rules).forEach(field => {
            rules[field].forEach(rule => {
                this.registerRule(field, rule);
            });
        });
        
        // Setup validation on submit
        form.addEventListener('submit', (e) => {
            const formData = new FormData(form);
            const validation = this.validateForm(formData);
            
            if (!validation.valid) {
                e.preventDefault();
                this.displayErrors(form, validation.errors);
                
                // Emit validation error event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('validation:failed', validation);
                }
            } else {
                // Emit validation success event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('validation:success', validation);
                }
            }
        });
        
        // Setup real-time validation
        form.querySelectorAll('input, textarea, select').forEach(input => {
            const field = input.name || input.id;
            if (field && this.rules.has(field)) {
                input.addEventListener('blur', () => {
                    const value = input.value;
                    const result = this.validateField(field, value);
                    this.displayFieldError(input, result);
                });
            }
        });
    },
    
    /**
     * Display errors
     */
    displayErrors(form, errors) {
        errors.forEach(({ field, errors: fieldErrors }) => {
            const input = form.querySelector(`[name="${field}"], #${field}`);
            if (input) {
                const errorMessages = fieldErrors.map(e => e.message).join(', ');
                this.displayFieldError(input, { valid: false, errors: fieldErrors });
            }
        });
    },
    
    /**
     * Display field error
     */
    displayFieldError(input, result) {
        // Remove existing error
        const existingError = input.parentElement.querySelector('.validation-error');
        if (existingError) {
            existingError.remove();
        }
        
        // Remove error class
        input.classList.remove('error');
        
        if (!result.valid && result.errors.length > 0) {
            // Add error class
            input.classList.add('error');
            
            // Create error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'validation-error';
            errorDiv.textContent = result.errors[0].message;
            input.parentElement.appendChild(errorDiv);
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    ValidationEngine.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ValidationEngine;
}

