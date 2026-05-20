import { useState, useCallback } from 'react';
import { z } from 'zod';

export interface UseFormOptions<T> {
  initialValues: T;
  validationSchema: z.ZodSchema<T>;
  onSubmit: (values: T) => Promise<void> | void;
}

export interface FormErrors {
  [key: string]: string | undefined;
}

export interface UseFormReturn<T> {
  values: T;
  errors: FormErrors;
  touched: { [key: string]: boolean };
  isSubmitting: boolean;
  isValid: boolean;
  setValue: <K extends keyof T>(field: K, value: T[K]) => void;
  setError: (field: keyof T, error: string) => void;
  handleChange: <K extends keyof T>(field: K) => (value: T[K]) => void;
  handleBlur: <K extends keyof T>(field: K) => () => void;
  handleSubmit: () => Promise<void>;
  reset: () => void;
}

export function useForm<T extends Record<string, any>>({
  initialValues,
  validationSchema,
  onSubmit,
}: UseFormOptions<T>): UseFormReturn<T> {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<{ [key: string]: boolean }>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validate = useCallback(
    (valuesToValidate: T): boolean => {
      try {
        validationSchema.parse(valuesToValidate);
        setErrors({});
        return true;
      } catch (error) {
        if (error instanceof z.ZodError) {
          const newErrors: FormErrors = {};
          error.errors.forEach((err) => {
            if (err.path.length > 0) {
              newErrors[err.path[0] as string] = err.message;
            }
          });
          setErrors(newErrors);
        }
        return false;
      }
    },
    [validationSchema]
  );

  const setValue = useCallback(<K extends keyof T>(field: K, value: T[K]) => {
    setValues((prev) => {
      const newValues = { ...prev, [field]: value };
      if (touched[field as string]) {
        validate(newValues);
      }
      return newValues;
    });
  }, [touched, validate]);

  const setError = useCallback((field: keyof T, error: string) => {
    setErrors((prev) => ({ ...prev, [field as string]: error }));
  }, []);

  const handleChange = useCallback(
    <K extends keyof T>(field: K) => {
      return (value: T[K]) => {
        setValue(field, value);
      };
    },
    [setValue]
  );

  const handleBlur = useCallback(
    <K extends keyof T>(field: K) => {
      return () => {
        setTouched((prev) => ({ ...prev, [field as string]: true }));
        validate(values);
      };
    },
    [values, validate]
  );

  const handleSubmit = useCallback(async () => {
    setTouched(
      Object.keys(values).reduce((acc, key) => {
        acc[key] = true;
        return acc;
      }, {} as { [key: string]: boolean })
    );

    if (!validate(values)) {
      return;
    }

    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } catch (error) {
      console.error('Form submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  }, [values, validate, onSubmit]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  const isValid = Object.keys(errors).length === 0 && Object.keys(touched).length > 0;

  return {
    values,
    errors,
    touched,
    isSubmitting,
    isValid,
    setValue,
    setError,
    handleChange,
    handleBlur,
    handleSubmit,
    reset,
  };
}


