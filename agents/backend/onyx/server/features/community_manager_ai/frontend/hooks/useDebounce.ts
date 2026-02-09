import { useState, useEffect } from 'react';
import { useDebounce as useDebounceHook } from 'use-debounce';

export const useDebounce = <T,>(value: T, delay: number = 300) => {
  const [debouncedValue] = useDebounceHook(value, delay);
  return debouncedValue;
};



