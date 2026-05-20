import { useState, useCallback } from 'react';

export const useFormState = <T extends Record<string, unknown>>(
  initialState: T
) => {
  const [state, setState] = useState<T>(initialState);

  const updateField = useCallback(<K extends keyof T>(field: K, value: T[K]): void => {
    setState((prev) => ({ ...prev, [field]: value }));
  }, []);

  const updateFields = useCallback((fields: Partial<T>): void => {
    setState((prev) => ({ ...prev, ...fields }));
  }, []);

  const reset = useCallback((): void => {
    setState(initialState);
  }, [initialState]);

  return {
    state,
    updateField,
    updateFields,
    reset,
    setState,
  };
};

