'use client';

import { useState, useCallback } from 'react';
import { formValidator, validators } from '@/lib/form-validator';

interface UseFormOptions<T> {
  initialValues: T;
  onSubmit: (values: T) => void | Promise<void>;
  validate?: (values: T) => Record<string, string[]>;
}

export function useForm<T extends Record<string, any>>({
  initialValues,
  onSubmit,
  validate,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Record<string, string[]>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const setValue = useCallback((field: keyof T, value: any) => {
    setValues((prev) => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field as string]) {
      setErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[field as string];
        return newErrors;
      });
    }
  }, [errors]);

  const setFieldTouched = useCallback((field: keyof T, isTouched: boolean = true) => {
    setTouched((prev) => ({ ...prev, [field]: isTouched }));
  }, []);

  const validateForm = useCallback((): boolean => {
    let validationErrors: Record<string, string[]> = {};

    if (validate) {
      const customErrors = validate(values);
      if (customErrors) {
        validationErrors = customErrors;
      }
    } else {
      const result = formValidator.validate(values);
      validationErrors = result.errors;
    }

    setErrors(validationErrors);
    return Object.keys(validationErrors).length === 0;
  }, [values, validate]);

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!validateForm()) {
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
  }, [values, onSubmit, validateForm]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  }, [initialValues]);

  const getFieldProps = useCallback((field: keyof T) => {
    return {
      value: values[field] ?? '',
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        setValue(field, e.target.value);
      },
      onBlur: () => {
        setFieldTouched(field, true);
        validateForm();
      },
      error: touched[field as string] ? errors[field as string]?.[0] : undefined,
    };
  }, [values, errors, touched, setValue, setFieldTouched, validateForm]);

  return {
    values,
    errors,
    touched,
    isSubmitting,
    setValue,
    setFieldTouched,
    validateForm,
    handleSubmit,
    reset,
    getFieldProps,
  };
}

export { validators };

