import { useState, useCallback } from 'react';

type ValidationRule<T> = {
  validator: (value: T) => boolean;
  message: string;
};

type FormField<T> = {
  value: T;
  error?: string;
  rules?: ValidationRule<T>[];
};

type FormFields = Record<string, FormField<any>>;

export const useForm = <T extends FormFields>(initialFields: T) => {
  const [fields, setFields] = useState<T>(initialFields);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const setValue = useCallback(
    <K extends keyof T>(name: K, value: T[K]['value']) => {
      setFields((prev) => ({
        ...prev,
        [name]: {
          ...prev[name],
          value,
          error: undefined,
        },
      }));
    },
    []
  );

  const setError = useCallback(
    <K extends keyof T>(name: K, error: string) => {
      setFields((prev) => ({
        ...prev,
        [name]: {
          ...prev[name],
          error,
        },
      }));
    },
    []
  );

  const validateField = useCallback(<K extends keyof T>(name: K): boolean => {
    const field = fields[name];
    if (!field.rules || field.rules.length === 0) return true;

    for (const rule of field.rules) {
      if (!rule.validator(field.value)) {
        setError(name, rule.message);
        return false;
      }
    }

    setError(name, undefined as any);
    return true;
  }, [fields, setError]);

  const validate = useCallback((): boolean => {
    let isValid = true;
    Object.keys(fields).forEach((key) => {
      if (!validateField(key as keyof T)) {
        isValid = false;
      }
    });
    return isValid;
  }, [fields, validateField]);

  const reset = useCallback(() => {
    setFields(initialFields);
    setIsSubmitting(false);
  }, [initialFields]);

  const getValues = useCallback((): Record<string, any> => {
    const values: Record<string, any> = {};
    Object.keys(fields).forEach((key) => {
      values[key] = fields[key as keyof T].value;
    });
    return values;
  }, [fields]);

  const handleSubmit = useCallback(
    async (onSubmit: (values: Record<string, any>) => Promise<void> | void) => {
      if (!validate()) {
        return;
      }

      setIsSubmitting(true);
      try {
        await onSubmit(getValues());
      } catch (error) {
        console.error('Form submission error:', error);
      } finally {
        setIsSubmitting(false);
      }
    },
    [validate, getValues]
  );

  return {
    fields,
    setValue,
    setError,
    validate,
    validateField,
    reset,
    getValues,
    handleSubmit,
    isSubmitting,
  };
};

