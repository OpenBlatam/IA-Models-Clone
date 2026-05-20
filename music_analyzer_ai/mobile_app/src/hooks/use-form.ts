import { useState, useCallback } from 'react';

interface ValidationRule<T> {
  validator: (value: T) => boolean;
  message: string;
}

interface FieldConfig<T> {
  initialValue: T;
  rules?: ValidationRule<T>[];
}

interface UseFormOptions<T extends Record<string, unknown>> {
  initialValues: T;
  validation?: Partial<Record<keyof T, ValidationRule<T[keyof T]>[]>>;
  onSubmit: (values: T) => void | Promise<void>;
}

/**
 * Hook for form management
 * Handles form state, validation, and submission
 */
export function useForm<T extends Record<string, unknown>>({
  initialValues,
  validation,
  onSubmit,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>(
    {}
  );
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validateField = useCallback(
    (name: keyof T, value: T[keyof T]): string | null => {
      const rules = validation?.[name];
      if (!rules) return null;

      for (const rule of rules) {
        if (!rule.validator(value)) {
          return rule.message;
        }
      }

      return null;
    },
    [validation]
  );

  const validateAll = useCallback((): boolean => {
    const newErrors: Partial<Record<keyof T, string>> = {};

    for (const key in values) {
      const error = validateField(key, values[key]);
      if (error) {
        newErrors[key] = error;
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [values, validateField]);

  const setValue = useCallback(
    (name: keyof T, value: T[keyof T]) => {
      setValues((prev) => ({ ...prev, [name]: value }));

      // Clear error when user starts typing
      if (errors[name]) {
        setErrors((prev) => {
          const newErrors = { ...prev };
          delete newErrors[name];
          return newErrors;
        });
      }

      // Validate on change if field was touched
      if (touched[name]) {
        const error = validateField(name, value);
        if (error) {
          setErrors((prev) => ({ ...prev, [name]: error }));
        }
      }
    },
    [errors, touched, validateField]
  );

  const setFieldTouched = useCallback((name: keyof T) => {
    setTouched((prev) => ({ ...prev, [name]: true }));

    // Validate on blur
    const error = validateField(name, values[name]);
    if (error) {
      setErrors((prev) => ({ ...prev, [name]: error }));
    }
  }, [values, validateField]);

  const handleSubmit = useCallback(async () => {
    // Mark all fields as touched
    const allTouched: Partial<Record<keyof T, boolean>> = {};
    for (const key in values) {
      allTouched[key] = true;
    }
    setTouched(allTouched);

    if (!validateAll()) {
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  }, [values, validateAll, onSubmit]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  return {
    values,
    errors,
    touched,
    isSubmitting,
    setValue,
    setFieldTouched,
    handleSubmit,
    reset,
    validateField,
    validateAll,
  };
}

