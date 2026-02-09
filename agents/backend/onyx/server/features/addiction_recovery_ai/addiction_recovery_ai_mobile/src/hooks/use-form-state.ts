import { useReducer, useCallback } from 'react';

interface FormState<T> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  isSubmitting: boolean;
}

type FormAction<T> =
  | { type: 'SET_VALUE'; field: keyof T; value: T[keyof T] }
  | { type: 'SET_ERROR'; field: keyof T; error: string }
  | { type: 'SET_TOUCHED'; field: keyof T; touched: boolean }
  | { type: 'SET_SUBMITTING'; isSubmitting: boolean }
  | { type: 'RESET'; initialValues: T }
  | { type: 'SET_ERRORS'; errors: Partial<Record<keyof T, string>> };

function formReducer<T>(state: FormState<T>, action: FormAction<T>): FormState<T> {
  switch (action.type) {
    case 'SET_VALUE':
      return {
        ...state,
        values: {
          ...state.values,
          [action.field]: action.value,
        },
        errors: {
          ...state.errors,
          [action.field]: undefined,
        },
      };
    case 'SET_ERROR':
      return {
        ...state,
        errors: {
          ...state.errors,
          [action.field]: action.error,
        },
      };
    case 'SET_TOUCHED':
      return {
        ...state,
        touched: {
          ...state.touched,
          [action.field]: action.touched,
        },
      };
    case 'SET_SUBMITTING':
      return {
        ...state,
        isSubmitting: action.isSubmitting,
      };
    case 'SET_ERRORS':
      return {
        ...state,
        errors: action.errors,
      };
    case 'RESET':
      return {
        values: action.initialValues,
        errors: {},
        touched: {},
        isSubmitting: false,
      };
    default:
      return state;
  }
}

export function useFormState<T extends Record<string, unknown>>(
  initialValues: T
) {
  const [state, dispatch] = useReducer(formReducer<T>, {
    values: initialValues,
    errors: {},
    touched: {},
    isSubmitting: false,
  });

  const setValue = useCallback((field: keyof T, value: T[keyof T]) => {
    dispatch({ type: 'SET_VALUE', field, value });
  }, []);

  const setError = useCallback((field: keyof T, error: string) => {
    dispatch({ type: 'SET_ERROR', field, error });
  }, []);

  const setTouched = useCallback((field: keyof T, touched: boolean = true) => {
    dispatch({ type: 'SET_TOUCHED', field, touched });
  }, []);

  const setErrors = useCallback((errors: Partial<Record<keyof T, string>>) => {
    dispatch({ type: 'SET_ERRORS', errors });
  }, []);

  const setSubmitting = useCallback((isSubmitting: boolean) => {
    dispatch({ type: 'SET_SUBMITTING', isSubmitting });
  }, []);

  const reset = useCallback(() => {
    dispatch({ type: 'RESET', initialValues });
  }, [initialValues]);

  const handleChange = useCallback(
    (field: keyof T) => (value: T[keyof T]) => {
      setValue(field, value);
    },
    [setValue]
  );

  const handleBlur = useCallback(
    (field: keyof T) => () => {
      setTouched(field, true);
    },
    [setTouched]
  );

  return {
    values: state.values,
    errors: state.errors,
    touched: state.touched,
    isSubmitting: state.isSubmitting,
    setValue,
    setError,
    setTouched,
    setErrors,
    setSubmitting,
    reset,
    handleChange,
    handleBlur,
  };
}

