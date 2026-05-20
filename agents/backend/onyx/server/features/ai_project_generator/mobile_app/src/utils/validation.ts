export const validators = {
  required: (value: string): string | null => {
    if (!value || value.trim().length === 0) {
      return 'Este campo es requerido';
    }
    return null;
  },

  minLength: (min: number) => (value: string): string | null => {
    if (value && value.length < min) {
      return `Debe tener al menos ${min} caracteres`;
    }
    return null;
  },

  maxLength: (max: number) => (value: string): string | null => {
    if (value && value.length > max) {
      return `No puede exceder ${max} caracteres`;
    }
    return null;
  },

  email: (value: string): string | null => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (value && !emailRegex.test(value)) {
      return 'Email inválido';
    }
    return null;
  },

  url: (value: string): string | null => {
    try {
      new URL(value);
      return null;
    } catch {
      return 'URL inválida';
    }
  },

  projectName: (value: string): string | null => {
    if (!value) return null;
    if (!/^[a-z0-9_]+$/.test(value)) {
      return 'Solo letras minúsculas, números y guiones bajos';
    }
    return null;
  },

  version: (value: string): string | null => {
    if (!value) return null;
    if (!/^\d+\.\d+\.\d+$/.test(value)) {
      return 'Formato inválido (debe ser x.y.z)';
    }
    return null;
  },

  number: (value: string): string | null => {
    if (value && isNaN(Number(value))) {
      return 'Debe ser un número';
    }
    return null;
  },

  positiveNumber: (value: string): string | null => {
    const num = Number(value);
    if (value && (isNaN(num) || num <= 0)) {
      return 'Debe ser un número positivo';
    }
    return null;
  },
};

export const validate = (
  value: string,
  rules: Array<(value: string) => string | null>
): string | null => {
  for (const rule of rules) {
    const error = rule(value);
    if (error) return error;
  }
  return null;
};

