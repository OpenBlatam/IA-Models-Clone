import { useState, useCallback } from 'react';
import { z } from 'zod';
import type { ZodSchema } from 'zod';

export function useFormValidation<T>(schema: ZodSchema<T>) {
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});

  const validate = useCallback(
    (data: unknown): data is T => {
      try {
        schema.parse(data);
        setErrors({});
        return true;
      } catch (error) {
        if (error instanceof z.ZodError) {
          const fieldErrors: Partial<Record<keyof T, string>> = {};
          error.errors.forEach((err) => {
            if (err.path[0]) {
              fieldErrors[err.path[0] as keyof T] = err.message;
            }
          });
          setErrors(fieldErrors);
        }
        return false;
      }
    },
    [schema]
  );

  const validateField = useCallback(
    (field: keyof T, value: unknown): boolean => {
      try {
        const fieldSchema = schema.shape?.[field as string];
        if (fieldSchema) {
          fieldSchema.parse(value);
          setErrors((prev) => {
            const newErrors = { ...prev };
            delete newErrors[field];
            return newErrors;
          });
          return true;
        }
        return true;
      } catch (error) {
        if (error instanceof z.ZodError) {
          setErrors((prev) => ({
            ...prev,
            [field]: error.errors[0]?.message || 'Invalid value',
          }));
        }
        return false;
      }
    },
    [schema]
  );

  const setFieldTouched = useCallback((field: keyof T, isTouched = true) => {
    setTouched((prev) => ({ ...prev, [field]: isTouched }));
  }, []);

  const reset = useCallback(() => {
    setErrors({});
    setTouched({});
  }, []);

  const getFieldError = useCallback(
    (field: keyof T): string | undefined => {
      return touched[field] ? errors[field] : undefined;
    },
    [errors, touched]
  );

  return {
    validate,
    validateField,
    setFieldTouched,
    getFieldError,
    errors,
    touched,
    reset,
    hasErrors: Object.keys(errors).length > 0,
  };
}

export function useFieldValidation<T>(schema: ZodSchema<T>, fieldName: keyof T) {
  const [error, setError] = useState<string | undefined>();
  const [isTouched, setIsTouched] = useState(false);

  const validate = useCallback(
    (value: unknown): boolean => {
      try {
        const fieldSchema = schema.shape?.[fieldName as string];
        if (fieldSchema) {
          fieldSchema.parse(value);
          setError(undefined);
          return true;
        }
        return true;
      } catch (err) {
        if (err instanceof z.ZodError) {
          setError(err.errors[0]?.message || 'Invalid value');
        }
        return false;
      }
    },
    [schema, fieldName]
  );

  const markAsTouched = useCallback(() => {
    setIsTouched(true);
  }, []);

  const reset = useCallback(() => {
    setError(undefined);
    setIsTouched(false);
  }, []);

  return {
    validate,
    markAsTouched,
    reset,
    error: isTouched ? error : undefined,
    isTouched,
  };
}


