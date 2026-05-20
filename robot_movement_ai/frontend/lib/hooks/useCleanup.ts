import { useEffect, useRef } from 'react';

/**
 * Hook for managing cleanup functions
 */
export function useCleanup() {
  const cleanupsRef = useRef<Array<() => void>>([]);

  const addCleanup = (cleanup: () => void) => {
    cleanupsRef.current.push(cleanup);
  };

  useEffect(() => {
    return () => {
      cleanupsRef.current.forEach((cleanup) => {
        try {
          cleanup();
        } catch (error) {
          console.error('Error in cleanup:', error);
        }
      });
      cleanupsRef.current = [];
    };
  }, []);

  return { addCleanup };
}



