/**
 * Custom hook for form validation using Zod.
 * Provides validation, error handling, and form state management.
 */

import { useState, useCallback, useMemo } from 'react';
import { z } from 'zod';
import { getErrorMessage } from '@/lib/errors';

/**
 * Options for form validation hook.
 */
export interface UseFormValidationOptions<T extends z.ZodType> {
  schema: T;
  initialValues?: Partial<z.infer<T>>;
  onSubmit?: (values: z.infer<T>) => Promise<void> | void;
  validateOnChange?: boolean;
  validateOnBlur?: boolean;
}

/**
 * Validation result.
 */
export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
}

/**
 * Return type for useFormValidation hook.
 */
export interface UseFormValidationReturn<T extends z.ZodType> {
  values: Partial<z.infer<T>>;
  errors: Record<string, string>;
  isValid: boolean;
  isSubmitting: boolean;
  touched: Record<string, boolean>;
  setValue: (field: keyof z.infer<T>, value: unknown) => void;
  setValues: (values: Partial<z.infer<T>>) => void;
  setError: (field: keyof z.infer<T>, error: string) => void;
  clearError: (field: keyof z.infer<T>) => void;
  clearErrors: () => void;
  validate: () => ValidationResult;
  validateField: (field: keyof z.infer<T>) => ValidationResult;
  handleSubmit: (e?: React.FormEvent) => Promise<void>;
  handleChange: (field: keyof z.infer<T>) => (value: unknown) => void;
  handleBlur: (field: keyof z.infer<T>) => () => void;
  reset: () => void;
}

/**
 * Custom hook for form validation with Zod.
 * @param options - Validation options
 * @returns Form validation state and handlers
 */
export function useFormValidation<T extends z.ZodType>(
  options: UseFormValidationOptions<T>
): UseFormValidationReturn<T> {
  const {
    schema,
    initialValues = {},
    onSubmit,
    validateOnChange = false,
    validateOnBlur = true,
  } = options;

  const [values, setValuesState] = useState<Partial<z.infer<T>>>(initialValues);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  /**
   * Validates a single field.
   */
  const validateField = useCallback(
    (field: keyof z.infer<T>): ValidationResult => {
      try {
        // Create a partial schema for the field
        const fieldSchema = schema.shape?.[field as string];
        if (!fieldSchema) {
          return { isValid: true, errors: {} };
        }

        const fieldValue = values[field];
        fieldSchema.parse(fieldValue);

        return { isValid: true, errors: {} };
      } catch (error) {
        if (error instanceof z.ZodError) {
          const fieldError = error.errors[0]?.message || 'Invalid value';
          return {
            isValid: false,
            errors: { [field as string]: fieldError },
          };
        }
        return {
          isValid: false,
          errors: { [field as string]: getErrorMessage(error) },
        };
      }
    },
    [schema, values]
  );

  /**
   * Validates all fields.
   */
  const validate = useCallback((): ValidationResult => {
    try {
      schema.parse(values);
      return { isValid: true, errors: {} };
    } catch (error) {
      if (error instanceof z.ZodError) {
        const validationErrors: Record<string, string> = {};
        error.errors.forEach((err) => {
          const path = err.path.join('.');
          validationErrors[path] = err.message;
        });
        return { isValid: false, errors: validationErrors };
      }
      return {
        isValid: false,
        errors: { _form: getErrorMessage(error) },
      };
    }
  }, [schema, values]);

  /**
   * Sets a single field value.
   */
  const setValue = useCallback(
    (field: keyof z.infer<T>, value: unknown) => {
      setValuesState((prev) => ({ ...prev, [field]: value }));

      // Clear error for this field
      setErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[field as string];
        return newErrors;
      });

      // Validate on change if enabled
      if (validateOnChange) {
        const result = validateField(field);
        if (!result.isValid) {
          setErrors((prev) => ({ ...prev, ...result.errors }));
        }
      }
    },
    [validateField, validateOnChange]
  );

  /**
   * Sets multiple field values.
   */
  const setValues = useCallback((newValues: Partial<z.infer<T>>) => {
    setValuesState((prev) => ({ ...prev, ...newValues }));
  }, []);

  /**
   * Sets an error for a field.
   */
  const setError = useCallback((field: keyof z.infer<T>, error: string) => {
    setErrors((prev) => ({ ...prev, [field as string]: error }));
  }, []);

  /**
   * Clears error for a field.
   */
  const clearError = useCallback((field: keyof z.infer<T>) => {
    setErrors((prev) => {
      const newErrors = { ...prev };
      delete newErrors[field as string];
      return newErrors;
    });
  }, []);

  /**
   * Clears all errors.
   */
  const clearErrors = useCallback(() => {
    setErrors({});
  }, []);

  /**
   * Handles field change.
   */
  const handleChange = useCallback(
    (field: keyof z.infer<T>) => (value: unknown) => {
      setValue(field, value);
    },
    [setValue]
  );

  /**
   * Handles field blur.
   */
  const handleBlur = useCallback(
    (field: keyof z.infer<T>) => () => {
      setTouched((prev) => ({ ...prev, [field as string]: true }));

      if (validateOnBlur) {
        const result = validateField(field);
        if (!result.isValid) {
          setErrors((prev) => ({ ...prev, ...result.errors }));
        }
      }
    },
    [validateField, validateOnBlur]
  );

  /**
   * Handles form submission.
   */
  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();

      setIsSubmitting(true);
      clearErrors();

      const validationResult = validate();
      if (!validationResult.isValid) {
        setErrors(validationResult.errors);
        setIsSubmitting(false);
        return;
      }

      try {
        if (onSubmit) {
          await onSubmit(values as z.infer<T>);
        }
      } catch (error) {
        const errorMessage = getErrorMessage(error);
        setError('_form' as keyof z.infer<T>, errorMessage);
      } finally {
        setIsSubmitting(false);
      }
    },
    [validate, onSubmit, values, clearErrors, setError]
  );

  /**
   * Resets form to initial values.
   */
  const reset = useCallback(() => {
    setValuesState(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  // Compute isValid from errors
  const isValid = useMemo(() => {
    return Object.keys(errors).length === 0 && validate().isValid;
  }, [errors, validate]);

  return {
    values,
    errors,
    isValid,
    isSubmitting,
    touched,
    setValue,
    setValues,
    setError,
    clearError,
    clearErrors,
    validate,
    validateField,
    handleSubmit,
    handleChange,
    handleBlur,
    reset,
  };
}
