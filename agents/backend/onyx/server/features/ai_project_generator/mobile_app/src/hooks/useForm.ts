import { useState, useCallback } from 'react';
import { validate, validators } from '../utils/validation';

export interface FormField {
  value: string;
  error: string | null;
  touched: boolean;
}

export interface UseFormOptions {
  initialValues: Record<string, string>;
  validationRules?: Record<string, Array<(value: string) => string | null>>;
  onSubmit: (values: Record<string, string>) => void | Promise<void>;
}

export const useForm = ({
  initialValues,
  validationRules = {},
  onSubmit,
}: UseFormOptions) => {
  const [fields, setFields] = useState<Record<string, FormField>>(() => {
    const initial: Record<string, FormField> = {};
    Object.keys(initialValues).forEach((key) => {
      initial[key] = {
        value: initialValues[key],
        error: null,
        touched: false,
      };
    });
    return initial;
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const setValue = useCallback((name: string, value: string) => {
    setFields((prev) => {
      const field = prev[name] || { value: '', error: null, touched: false };
      const rules = validationRules[name] || [];
      const error = validate(value, rules);

      return {
        ...prev,
        [name]: {
          ...field,
          value,
          error,
          touched: true,
        },
      };
    });
  }, [validationRules]);

  const setError = useCallback((name: string, error: string | null) => {
    setFields((prev) => ({
      ...prev,
      [name]: {
        ...prev[name],
        error,
        touched: true,
      },
    }));
  }, []);

  const validateForm = useCallback((): boolean => {
    let isValid = true;
    const newFields = { ...fields };

    Object.keys(fields).forEach((key) => {
      const field = fields[key];
      const rules = validationRules[key] || [];
      const error = validate(field.value, rules);

      if (error) {
        isValid = false;
        newFields[key] = {
          ...field,
          error,
          touched: true,
        };
      }
    });

    setFields(newFields);
    return isValid;
  }, [fields, validationRules]);

  const handleSubmit = useCallback(async () => {
    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    try {
      const values: Record<string, string> = {};
      Object.keys(fields).forEach((key) => {
        values[key] = fields[key].value;
      });
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  }, [fields, validateForm, onSubmit]);

  const reset = useCallback(() => {
    const resetFields: Record<string, FormField> = {};
    Object.keys(initialValues).forEach((key) => {
      resetFields[key] = {
        value: initialValues[key],
        error: null,
        touched: false,
      };
    });
    setFields(resetFields);
  }, [initialValues]);

  const getFieldProps = useCallback(
    (name: string) => ({
      value: fields[name]?.value || '',
      onChangeText: (text: string) => setValue(name, text),
      error: fields[name]?.error,
      touched: fields[name]?.touched || false,
    }),
    [fields, setValue]
  );

  return {
    fields,
    setValue,
    setError,
    validateForm,
    handleSubmit,
    reset,
    getFieldProps,
    isSubmitting,
    isValid: Object.values(fields).every((f) => !f.error),
  };
};

export { validators };

