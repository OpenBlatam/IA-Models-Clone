/**
 * Custom hook for form state management.
 * Provides convenient form state handling with validation.
 */

import { useState, useCallback, type FormEvent } from 'react';

/**
 * Form state interface.
 */
export interface FormState<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
  isValid: boolean;
}

/**
 * Options for useForm hook.
 */
export interface UseFormOptions<T> {
  initialValues: T;
  validate?: (values: T) => Partial<Record<keyof T, string>>;
  onSubmit: (values: T) => void | Promise<void>;
}

/**
 * Return type for useForm hook.
 */
export interface UseFormReturn<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
  isValid: boolean;
  setValue: <K extends keyof T>(name: K, value: T[K]) => void;
  setError: (name: keyof T, error: string) => void;
  setTouched: (name: keyof T, touched: boolean) => void;
  handleChange: <K extends keyof T>(
    name: K
  ) => (value: T[K] | React.ChangeEvent<HTMLInputElement>) => void;
  handleBlur: (name: keyof T) => () => void;
  handleSubmit: (e: FormEvent) => Promise<void>;
  reset: () => void;
}

/**
 * Custom hook for form state management.
 * Provides convenient form handling with validation.
 *
 * @param options - Hook options
 * @returns Form state and handlers
 */
export function useForm<T extends Record<string, any>>(
  options: UseFormOptions<T>
): UseFormReturn<T> {
  const { initialValues, validate, onSubmit } = options;

  const [state, setState] = useState<FormState<T>>(() => {
    const errors = validate ? validate(initialValues) : {};
    return {
      values: initialValues,
      errors,
      touched: {},
      isSubmitting: false,
      isValid: Object.keys(errors).length === 0,
    };
  });

  const setValue = useCallback(<K extends keyof T>(name: K, value: T[K]) => {
    setState((prev) => {
      const newValues = { ...prev.values, [name]: value };
      const errors = validate ? validate(newValues) : {};
      return {
        ...prev,
        values: newValues,
        errors,
        isValid: Object.keys(errors).length === 0,
      };
    });
  }, [validate]);

  const setError = useCallback((name: keyof T, error: string) => {
    setState((prev) => ({
      ...prev,
      errors: { ...prev.errors, [name]: error },
      isValid: false,
    }));
  }, []);

  const setTouched = useCallback((name: keyof T, touched: boolean) => {
    setState((prev) => ({
      ...prev,
      touched: { ...prev.touched, [name]: touched },
    }));
  }, []);

  const handleChange = useCallback(
    <K extends keyof T>(name: K) => {
      return (value: T[K] | React.ChangeEvent<HTMLInputElement>) => {
        const actualValue =
          typeof value === 'object' && 'target' in value
            ? (value.target as HTMLInputElement).value
            : value;
        setValue(name, actualValue as T[K]);
      };
    },
    [setValue]
  );

  const handleBlur = useCallback(
    (name: keyof T) => () => {
      setTouched(name, true);
    },
    [setTouched]
  );

  const handleSubmit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();

      if (!state.isValid) {
        // Mark all fields as touched
        const allTouched = Object.keys(state.values).reduce(
          (acc, key) => ({ ...acc, [key]: true }),
          {} as Partial<Record<keyof T, boolean>>
        );
        setState((prev) => ({ ...prev, touched: allTouched }));
        return;
      }

      setState((prev) => ({ ...prev, isSubmitting: true }));

      try {
        await onSubmit(state.values);
      } finally {
        setState((prev) => ({ ...prev, isSubmitting: false }));
      }
    },
    [state.values, state.isValid, onSubmit]
  );

  const reset = useCallback(() => {
    const errors = validate ? validate(initialValues) : {};
    setState({
      values: initialValues,
      errors,
      touched: {},
      isSubmitting: false,
      isValid: Object.keys(errors).length === 0,
    });
  }, [initialValues, validate]);

  return {
    values: state.values,
    errors: state.errors,
    touched: state.touched,
    isSubmitting: state.isSubmitting,
    isValid: state.isValid,
    setValue,
    setError,
    setTouched,
    handleChange,
    handleBlur,
    handleSubmit,
    reset,
  };
}

