export const VALIDATION_RULES = {
  title: {
    minLength: 3,
    maxLength: 200,
  },
  description: {
    minLength: 10,
    maxLength: 2000,
  },
  priority: {
    min: 1,
    max: 10,
  },
  duration: {
    min: 1,
    max: 1440, // 24 hours in minutes
  },
  email: {
    pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  },
  url: {
    pattern: /^https?:\/\/.+/,
  },
} as const;

export const ERROR_MESSAGES = {
  required: 'Este campo es requerido',
  minLength: (min: number) => `Debe tener al menos ${min} caracteres`,
  maxLength: (max: number) => `No puede exceder ${max} caracteres`,
  min: (min: number) => `El valor mínimo es ${min}`,
  max: (max: number) => `El valor máximo es ${max}`,
  email: 'Debe ser un email válido',
  url: 'Debe ser una URL válida',
  invalidDate: 'Fecha inválida',
  invalidTime: 'Hora inválida',
} as const;

