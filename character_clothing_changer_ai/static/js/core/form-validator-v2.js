/**
 * Form Validator V2 Module
 * ========================
 * Advanced form validation with real-time feedback
 */

const FormValidatorV2 = {
    /**
     * Validation rules
     */
    rules: new Map(),
    
    /**
     * Validators
     */
    validators: new Map(),
    
    /**
     * Form states
     */
    formStates: new Map(),
    
    /**
     * Initialize form validator
     */
    init() {
        // Register default validators
        this.registerDefaultValidators();
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Form validator V2 initialized');
        }
    },
    
    /**
     * Register default validators
     */
    registerDefaultValidators() {
        // Required
        this.registerValidator('required', (value) => {
            if (typeof value === 'string') {
                return value.trim().length > 0;
            }
            return value !== null && value !== undefined;
        }, 'Este campo es obligatorio');
        
        // Email
        this.registerValidator('email', (value) => {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(value);
        }, 'Ingresa un email válido');
        
        // Min length
        this.registerValidator('minLength', (value, min) => {
            return value.length >= min;
        }, (min) => `Mínimo ${min} caracteres`);
        
        // Max length
        this.registerValidator('maxLength', (value, max) => {
            return value.length <= max;
        }, (max) => `Máximo ${max} caracteres`);
        
        // Pattern
        this.registerValidator('pattern', (value, pattern) => {
            const regex = new RegExp(pattern);
            return regex.test(value);
        }, 'El formato no es válido');
        
        // Number
        this.registerValidator('number', (value) => {
            return !isNaN(value) && !isNaN(parseFloat(value));
        }, 'Debe ser un número');
        
        // Min value
        this.registerValidator('min', (value, min) => {
            return parseFloat(value) >= min;
        }, (min) => `El valor mínimo es ${min}`);
        
        // Max value
        this.registerValidator('max', (value, max) => {
            return parseFloat(value) <= max;
        }, (max) => `El valor máximo es ${max}`);
        
        // File type
        this.registerValidator('fileType', (file, types) => {
            if (!file) return false;
            const fileType = file.type || file.name.split('.').pop();
            return types.includes(fileType);
        }, 'Tipo de archivo no válido');
        
        // File size
        this.registerValidator('fileSize', (file, maxSize) => {
            if (!file) return false;
            return file.size <= maxSize;
        }, (maxSize) => `El archivo no debe exceder ${maxSize / 1024 / 1024}MB`);
    },
    
    /**
     * Register validator
     */
    registerValidator(name, validatorFn, errorMessage) {
        this.validators.set(name, {
            validate: validatorFn,
            errorMessage: typeof errorMessage === 'function' ? errorMessage : () => errorMessage
        });
    },
    
    /**
     * Validate field
     */
    validateField(field, value, rules) {
        const errors = [];
        
        for (const rule of rules) {
            const [ruleName, ...ruleParams] = rule.split(':');
            const validator = this.validators.get(ruleName);
            
            if (!validator) {
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Validator not found: ${ruleName}`);
                }
                continue;
            }
            
            try {
                const isValid = validator.validate(value, ...ruleParams);
                if (!isValid) {
                    const errorMsg = validator.errorMessage(...ruleParams);
                    errors.push(errorMsg);
                }
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error('Validation error', error);
                }
                errors.push('Error de validación');
            }
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    },
    
    /**
     * Validate form
     */
    validateForm(formElement) {
        const formData = new FormData(formElement);
        const errors = {};
        let isValid = true;
        
        // Get all fields with validation rules
        const fields = formElement.querySelectorAll('[data-validate]');
        
        fields.forEach(field => {
            const rules = field.getAttribute('data-validate').split('|');
            const value = field.value || formData.get(field.name);
            
            const result = this.validateField(field.name, value, rules);
            
            if (!result.valid) {
                errors[field.name] = result.errors;
                isValid = false;
            }
        });
        
        return {
            valid: isValid,
            errors
        };
    },
    
    /**
     * Setup real-time validation
     */
    setupRealTimeValidation(formElement) {
        const fields = formElement.querySelectorAll('[data-validate]');
        
        fields.forEach(field => {
            // Validate on blur
            field.addEventListener('blur', () => {
                this.validateFieldRealTime(field);
            });
            
            // Validate on input (debounced)
            if (typeof debounce !== 'undefined') {
                field.addEventListener('input', debounce(() => {
                    this.validateFieldRealTime(field);
                }, 300));
            } else {
                field.addEventListener('input', () => {
                    this.validateFieldRealTime(field);
                });
            }
        });
    },
    
    /**
     * Validate field real-time
     */
    validateFieldRealTime(field) {
        const rules = field.getAttribute('data-validate').split('|');
        const value = field.value;
        
        const result = this.validateField(field.name, value, rules);
        
        // Update UI
        this.updateFieldValidation(field, result);
    },
    
    /**
     * Update field validation UI
     */
    updateFieldValidation(field, result) {
        // Remove existing error messages
        const existingError = field.parentElement.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }
        
        // Remove error classes
        field.classList.remove('error', 'valid');
        
        if (result.valid) {
            field.classList.add('valid');
        } else {
            field.classList.add('error');
            
            // Add error message
            const errorMsg = document.createElement('div');
            errorMsg.className = 'error-message';
            errorMsg.textContent = result.errors[0];
            field.parentElement.appendChild(errorMsg);
        }
    },
    
    /**
     * Clear validation
     */
    clearValidation(formElement) {
        const fields = formElement.querySelectorAll('[data-validate]');
        fields.forEach(field => {
            field.classList.remove('error', 'valid');
            const errorMsg = field.parentElement.querySelector('.error-message');
            if (errorMsg) {
                errorMsg.remove();
            }
        });
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    FormValidatorV2.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FormValidatorV2;
}

