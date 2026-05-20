import { useCallback } from 'react';

export interface UseFieldOptions {
  validate?: (value: any) => string | null;
  onChange?: (value: any) => void;
  onBlur?: () => void;
}

export function useField(
  name: string,
  value: any,
  setValue: (value: any) => void,
  error?: string,
  touched?: boolean,
  options: UseFieldOptions = {}
) {
  const { validate, onChange, onBlur } = options;

  const handleChange = useCallback(
    (newValue: any) => {
      setValue(newValue);
      if (onChange) {
        onChange(newValue);
      }
    },
    [setValue, onChange]
  );

  const handleBlur = useCallback(() => {
    if (onBlur) {
      onBlur();
    }
  }, [onBlur]);

  const validateField = useCallback((): string | null => {
    if (validate) {
      return validate(value);
    }
    return null;
  }, [value, validate]);

  return {
    name,
    value,
    error,
    touched,
    onChange: handleChange,
    onBlur: handleBlur,
    validate: validateField,
  };
}



