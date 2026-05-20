/**
 * Form Utility Functions
 * Utility functions for form handling and validation
 */

import { FieldErrors, FieldValues } from 'react-hook-form';

/**
 * Gets the first error message from form errors
 * @param errors - Form errors object
 * @returns First error message or undefined
 */
export const getFirstError = (errors: FieldErrors<FieldValues>): string | undefined => {
  const firstError = Object.values(errors)[0];
  if (!firstError) return undefined;
  
  if (typeof firstError === 'object' && 'message' in firstError) {
    return firstError.message as string;
  }
  
  return undefined;
};

/**
 * Checks if form has any errors
 * @param errors - Form errors object
 * @returns True if form has errors
 */
export const hasFormErrors = (errors: FieldErrors<FieldValues>): boolean => {
  return Object.keys(errors).length > 0;
};

/**
 * Gets all error messages from form errors
 * @param errors - Form errors object
 * @returns Array of error messages
 */
export const getAllErrors = (errors: FieldErrors<FieldValues>): string[] => {
  const errorMessages: string[] = [];
  
  const extractErrors = (obj: unknown, prefix = ''): void => {
    if (typeof obj !== 'object' || obj === null) return;
    
    Object.entries(obj).forEach(([key, value]) => {
      const fieldPath = prefix ? `${prefix}.${key}` : key;
      
      if (typeof value === 'object' && value !== null) {
        if ('message' in value && typeof value.message === 'string') {
          errorMessages.push(value.message);
        } else {
          extractErrors(value, fieldPath);
        }
      }
    });
  };
  
  extractErrors(errors);
  return errorMessages;
};

/**
 * Formats form data for submission
 * @param data - Form data object
 * @param excludeEmpty - Whether to exclude empty values (default: true)
 * @returns Formatted form data
 */
export const formatFormData = <T extends Record<string, unknown>>(
  data: T,
  excludeEmpty: boolean = true
): Partial<T> => {
  const formatted: Partial<T> = {};
  
  Object.entries(data).forEach(([key, value]) => {
    if (!excludeEmpty || (value !== '' && value !== null && value !== undefined)) {
      formatted[key as keyof T] = value as T[keyof T];
    }
  });
  
  return formatted;
};

/**
 * Resets form to initial values
 * @param reset - React Hook Form reset function
 * @param defaultValues - Default values to reset to
 */
export const resetForm = <T extends FieldValues>(
  reset: (values?: T) => void,
  defaultValues?: T
): void => {
  reset(defaultValues || ({} as T));
};

/**
 * Validates file size
 * @param file - File to validate
 * @param maxSize - Maximum size in bytes
 * @returns Error message or undefined
 */
export const validateFileSize = (file: File, maxSize: number): string | undefined => {
  if (file.size > maxSize) {
    const maxSizeMB = (maxSize / 1024 / 1024).toFixed(2);
    return `El archivo es demasiado grande. Tamaño máximo: ${maxSizeMB}MB`;
  }
  return undefined;
};

/**
 * Validates file type
 * @param file - File to validate
 * @param allowedTypes - Array of allowed MIME types
 * @returns Error message or undefined
 */
export const validateFileType = (file: File, allowedTypes: string[]): string | undefined => {
  const isValid = allowedTypes.some((type) => {
    if (type.endsWith('/*')) {
      const baseType = type.split('/')[0];
      return file.type.startsWith(`${baseType}/`);
    }
    return file.type === type;
  });
  
  if (!isValid) {
    return `Tipo de archivo no permitido. Tipos permitidos: ${allowedTypes.join(', ')}`;
  }
  return undefined;
};

/**
 * Creates FormData from object
 * @param data - Data object
 * @param excludeEmpty - Whether to exclude empty values (default: true)
 * @returns FormData instance
 */
export const createFormData = <T extends Record<string, unknown>>(
  data: T,
  excludeEmpty: boolean = true
): FormData => {
  const formData = new FormData();
  
  Object.entries(data).forEach(([key, value]) => {
    if (excludeEmpty && (value === '' || value === null || value === undefined)) {
      return;
    }
    
    if (value instanceof File || value instanceof FileList) {
      if (value instanceof FileList) {
        Array.from(value).forEach((file) => {
          formData.append(key, file);
        });
      } else {
        formData.append(key, value);
      }
    } else if (typeof value === 'object' && value !== null) {
      formData.append(key, JSON.stringify(value));
    } else {
      formData.append(key, String(value));
    }
  });
  
  return formData;
};

/**
 * Parses tags string to array
 * @param tags - Tags string (comma-separated)
 * @returns Array of trimmed tags
 */
export const parseTags = (tags: string | string[] | undefined): string[] => {
  if (!tags) return [];
  if (Array.isArray(tags)) return tags.filter(Boolean);
  return tags.split(',').map((tag) => tag.trim()).filter(Boolean);
};


