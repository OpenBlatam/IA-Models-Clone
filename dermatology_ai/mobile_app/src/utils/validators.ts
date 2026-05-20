/**
 * Common validation functions
 */

export const validators = {
  required: (message = 'Este campo es requerido') => ({
    validator: (value: any) => {
      if (typeof value === 'string') return value.trim().length > 0;
      if (Array.isArray(value)) return value.length > 0;
      return value !== null && value !== undefined;
    },
    message,
  }),

  email: (message = 'Email inválido') => ({
    validator: (value: string) => {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(value);
    },
    message,
  }),

  minLength: (min: number, message?: string) => ({
    validator: (value: string) => value.length >= min,
    message: message || `Mínimo ${min} caracteres`,
  }),

  maxLength: (max: number, message?: string) => ({
    validator: (value: string) => value.length <= max,
    message: message || `Máximo ${max} caracteres`,
  }),

  min: (min: number, message?: string) => ({
    validator: (value: number) => value >= min,
    message: message || `Valor mínimo: ${min}`,
  }),

  max: (max: number, message?: string) => ({
    validator: (value: number) => value <= max,
    message: message || `Valor máximo: ${max}`,
  }),

  pattern: (regex: RegExp, message: string) => ({
    validator: (value: string) => regex.test(value),
    message,
  }),

  phone: (message = 'Teléfono inválido') => ({
    validator: (value: string) => {
      const phoneRegex = /^[+]?[(]?[0-9]{1,4}[)]?[-\s.]?[(]?[0-9]{1,4}[)]?[-\s.]?[0-9]{1,9}$/;
      return phoneRegex.test(value);
    },
    message,
  }),

  url: (message = 'URL inválida') => ({
    validator: (value: string) => {
      try {
        new URL(value);
        return true;
      } catch {
        return false;
      }
    },
    message,
  }),

  match: (otherValue: any, message = 'Los valores no coinciden') => ({
    validator: (value: any) => value === otherValue,
    message,
  }),
};

