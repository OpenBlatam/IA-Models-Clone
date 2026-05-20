import { useCallback } from 'react';

export const usePrint = () => {
  const print = useCallback(() => {
    window.print();
  }, []);

  return { print };
};

