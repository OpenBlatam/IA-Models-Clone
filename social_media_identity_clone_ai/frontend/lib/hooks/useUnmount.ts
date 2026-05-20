import { useEffect, useRef } from 'react';

export const useUnmount = (callback: () => void): void => {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    return () => {
      callbackRef.current();
    };
  }, []);
};



