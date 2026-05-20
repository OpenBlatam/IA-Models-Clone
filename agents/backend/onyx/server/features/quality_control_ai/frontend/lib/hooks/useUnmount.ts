import { useEffect, useRef } from 'react';

export const useUnmount = (fn: () => void): void => {
  const fnRef = useRef(fn);

  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  useEffect(() => {
    return () => {
      fnRef.current();
    };
  }, []);
};
