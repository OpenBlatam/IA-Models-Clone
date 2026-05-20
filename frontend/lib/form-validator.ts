type ValidationRule = (value: any) => string | null;
type ValidationRules = Record<string, ValidationRule[]>;

interface ValidationResult {
  valid: boolean;
  errors: Record<string, string[]>;
}

class FormValidator {
  private rules: ValidationRules = {};

  addRule(field: string, rule: ValidationRule) {
    if (!this.rules[field]) {
      this.rules[field] = [];
    }
    this.rules[field].push(rule);
    return this;
  }

  removeRule(field: string, ruleIndex: number) {
    if (this.rules[field]) {
      this.rules[field].splice(ruleIndex, 1);
    }
    return this;
  }

  validateField(field: string, value: any): string[] {
    const errors: string[] = [];
    const fieldRules = this.rules[field] || [];

    for (const rule of fieldRules) {
      const error = rule(value);
      if (error) {
        errors.push(error);
      }
    }

    return errors;
  }

  validate(data: Record<string, any>): ValidationResult {
    const errors: Record<string, string[]> = {};

    for (const field in this.rules) {
      const fieldErrors = this.validateField(field, data[field]);
      if (fieldErrors.length > 0) {
        errors[field] = fieldErrors;
      }
    }

    return {
      valid: Object.keys(errors).length === 0,
      errors,
    };
  }

  clear() {
    this.rules = {};
    return this;
  }
}

// Built-in validation rules
export const validators = {
  required: (message: string = 'Este campo es requerido'): ValidationRule => {
    return (value: any) => {
      if (value === null || value === undefined || value === '') {
        return message;
      }
      if (Array.isArray(value) && value.length === 0) {
        return message;
      }
      return null;
    };
  },

  minLength: (min: number, message?: string): ValidationRule => {
    return (value: any) => {
      if (typeof value === 'string' && value.length < min) {
        return message || `Debe tener al menos ${min} caracteres`;
      }
      return null;
    };
  },

  maxLength: (max: number, message?: string): ValidationRule => {
    return (value: any) => {
      if (typeof value === 'string' && value.length > max) {
        return message || `Debe tener máximo ${max} caracteres`;
      }
      return null;
    };
  },

  email: (message: string = 'Email inválido'): ValidationRule => {
    return (value: any) => {
      if (!value) return null;
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(value)) {
        return message;
      }
      return null;
    };
  },

  url: (message: string = 'URL inválida'): ValidationRule => {
    return (value: any) => {
      if (!value) return null;
      try {
        new URL(value);
        return null;
      } catch {
        return message;
      }
    };
  },

  number: (message: string = 'Debe ser un número'): ValidationRule => {
    return (value: any) => {
      if (value === null || value === undefined || value === '') return null;
      if (isNaN(Number(value))) {
        return message;
      }
      return null;
    };
  },

  min: (min: number, message?: string): ValidationRule => {
    return (value: any) => {
      const num = Number(value);
      if (!isNaN(num) && num < min) {
        return message || `Debe ser mayor o igual a ${min}`;
      }
      return null;
    };
  },

  max: (max: number, message?: string): ValidationRule => {
    return (value: any) => {
      const num = Number(value);
      if (!isNaN(num) && num > max) {
        return message || `Debe ser menor o igual a ${max}`;
      }
      return null;
    };
  },

  pattern: (regex: RegExp, message: string = 'Formato inválido'): ValidationRule => {
    return (value: any) => {
      if (!value) return null;
      if (!regex.test(value)) {
        return message;
      }
      return null;
    };
  },

  custom: (fn: (value: any) => string | null): ValidationRule => {
    return fn;
  },
};

export const formValidator = new FormValidator();

