import { useState, useCallback } from 'react';

export interface UseFormStateOptions<T> {
  initialValues: T;
  onSubmit: (values: T) => Promise<void> | void;
  validate?: (values: T) => Partial<Record<keyof T, string>>;
  onSuccess?: (values: T) => void;
  onError?: (error: unknown) => void;
}

export function useFormState<T extends Record<string, any>>(
  options: UseFormStateOptions<T>
) {
  const { initialValues, onSubmit, validate, onSuccess, onError } = options;

  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});

  const setValue = useCallback(<K extends keyof T>(key: K, value: T[K]) => {
    setValues((prev) => ({ ...prev, [key]: value }));
    // Clear error when value changes
    if (errors[key]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }, [errors]);

  const setFieldError = useCallback(<K extends keyof T>(key: K, error: string) => {
    setErrors((prev) => ({ ...prev, [key]: error }));
  }, []);

  const setFieldTouched = useCallback(<K extends keyof T>(key: K, isTouched: boolean = true) => {
    setTouched((prev) => ({ ...prev, [key]: isTouched }));
  }, []);

  const validateForm = useCallback((): boolean => {
    if (validate) {
      const validationErrors = validate(values);
      setErrors(validationErrors);
      return Object.keys(validationErrors).length === 0;
    }
    return true;
  }, [values, validate]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault();
    }

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    setErrors({});

    try {
      await onSubmit(values);
      if (onSuccess) {
        onSuccess(values);
      }
    } catch (error) {
      if (onError) {
        onError(error);
      } else {
        setErrors({ _form: 'Ha ocurrido un error al enviar el formulario' } as any);
      }
    } finally {
      setIsSubmitting(false);
    }
  }, [values, validateForm, onSubmit, onSuccess, onError]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
    setIsSubmitting(false);
  }, [initialValues]);

  const resetField = useCallback(<K extends keyof T>(key: K) => {
    setValue(key, initialValues[key]);
    setFieldError(key, '');
    setFieldTouched(key, false);
  }, [initialValues, setValue, setFieldError, setFieldTouched]);

  return {
    values,
    errors,
    touched,
    isSubmitting,
    setValue,
    setValues,
    setFieldError,
    setFieldTouched,
    handleSubmit,
    validate: validateForm,
    reset,
    resetField,
  };
}



