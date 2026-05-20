import { useState, useCallback } from 'react';

export const useToggleState = (initialState = false) => {
  const [state, setState] = useState(initialState);

  const toggle = useCallback(() => {
    setState((prev) => !prev);
  }, []);

  const setTrue = useCallback(() => {
    setState(true);
  }, []);

  const setFalse = useCallback(() => {
    setState(false);
  }, []);

  const set = useCallback((value: boolean) => {
    setState(value);
  }, []);

  return [state, { toggle, setTrue, setFalse, set }] as const;
};

