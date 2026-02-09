import { useState, useCallback } from 'react';

export const useToggle = (initialValue = false): [boolean, () => void, (value: boolean) => void] => {
  const [value, setValue] = useState(initialValue);

  const toggle = useCallback((): void => {
    setValue((prev) => !prev);
  }, []);

  const setToggle = useCallback((newValue: boolean): void => {
    setValue(newValue);
  }, []);

  return [value, toggle, setToggle];
};



