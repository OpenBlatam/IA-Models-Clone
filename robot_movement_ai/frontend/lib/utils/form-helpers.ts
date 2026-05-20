/**
 * Form helper utilities
 */

import { handleValidationError } from './error-handler';
import { toast } from './toast';
import { SUCCESS_MESSAGES } from './constants';

export interface FormSubmitOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: unknown) => void;
  successMessage?: string;
  showSuccessToast?: boolean;
}

/**
 * Handle form submission with error handling
 */
export async function handleFormSubmit<T>(
  submitFn: () => Promise<T>,
  options: FormSubmitOptions = {}
): Promise<T | null> {
  const {
    onSuccess,
    onError,
    successMessage = SUCCESS_MESSAGES.SAVED,
    showSuccessToast = true,
  } = options;

  try {
    const data = await submitFn();

    if (showSuccessToast) {
      toast.success(successMessage);
    }

    if (onSuccess) {
      onSuccess(data);
    }

    return data;
  } catch (error) {
    handleValidationError(error);

    if (onError) {
      onError(error);
    }

    return null;
  }
}

/**
 * Create form submit handler
 */
export function createFormSubmitHandler<T>(
  submitFn: () => Promise<T>,
  options?: FormSubmitOptions
) {
  return () => handleFormSubmit(submitFn, options);
}

/**
 * Validate form field
 */
export function validateField(
  value: any,
  rules: Array<(val: any) => string | null>
): string | null {
  for (const rule of rules) {
    const error = rule(value);
    if (error) {
      return error;
    }
  }
  return null;
}

/**
 * Common validation rules
 */
export const validationRules = {
  required: (message: string = 'Este campo es requerido') => (value: any) => {
    if (value === null || value === undefined || value === '') {
      return message;
    }
    return null;
  },
  minLength: (min: number, message?: string) => (value: string) => {
    if (value.length < min) {
      return message || `Debe tener al menos ${min} caracteres`;
    }
    return null;
  },
  maxLength: (max: number, message?: string) => (value: string) => {
    if (value.length > max) {
      return message || `Debe tener máximo ${max} caracteres`;
    }
    return null;
  },
  email: (message: string = 'Email inválido') => (value: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      return message;
    }
    return null;
  },
  number: (message: string = 'Debe ser un número') => (value: any) => {
    if (isNaN(Number(value))) {
      return message;
    }
    return null;
  },
  min: (min: number, message?: string) => (value: number) => {
    if (value < min) {
      return message || `Debe ser mayor o igual a ${min}`;
    }
    return null;
  },
  max: (max: number, message?: string) => (value: number) => {
    if (value > max) {
      return message || `Debe ser menor o igual a ${max}`;
    }
    return null;
  },
};



