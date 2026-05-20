import { useState, useRef, useEffect, RefObject } from 'react';

export function useFocus<T extends HTMLElement = HTMLElement>(): [RefObject<T>, boolean, () => void] {
  const [isFocused, setIsFocused] = useState(false);
  const ref = useRef<T>(null);

  const focus = () => {
    ref.current?.focus();
  };

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const handleFocus = () => setIsFocused(true);
    const handleBlur = () => setIsFocused(false);

    element.addEventListener('focus', handleFocus);
    element.addEventListener('blur', handleBlur);

    return () => {
      element.removeEventListener('focus', handleFocus);
      element.removeEventListener('blur', handleBlur);
    };
  }, []);

  return [ref, isFocused, focus];
}

