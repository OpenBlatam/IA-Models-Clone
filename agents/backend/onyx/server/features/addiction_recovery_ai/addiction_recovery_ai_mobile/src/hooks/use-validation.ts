import { useState, useCallback } from 'react';
import { z } from 'zod';
import { validateWithSchema, getValidationErrors } from '@/utils/validation-helpers';

export function useValidation<T>(schema: z.ZodSchema<T>) {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isValid, setIsValid] = useState(false);

  const validate = useCallback(
    (data: unknown): data is T => {
      const result = validateWithSchema(schema, data);

      if (result.success) {
        setErrors({});
        setIsValid(true);
        return true;
      }

      const validationErrors = getValidationErrors(result.errors);
      setErrors(validationErrors);
      setIsValid(false);
      return false;
    },
    [schema]
  );

  const clearErrors = useCallback(() => {
    setErrors({});
    setIsValid(false);
  }, []);

  const getError = useCallback(
    (field: string): string | undefined => {
      return errors[field];
    },
    [errors]
  );

  return {
    validate,
    errors,
    isValid,
    clearErrors,
    getError,
  };
}

