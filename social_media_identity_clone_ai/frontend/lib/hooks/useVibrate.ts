import { useCallback } from 'react';

export const useVibrate = () => {
  const vibrate = useCallback((pattern: number | number[]) => {
    if (typeof window === 'undefined' || !('vibrate' in navigator)) {
      return false;
    }

    try {
      navigator.vibrate(pattern);
      return true;
    } catch {
      return false;
    }
  }, []);

  const stop = useCallback(() => {
    if (typeof window !== 'undefined' && 'vibrate' in navigator) {
      navigator.vibrate(0);
    }
  }, []);

  return {
    vibrate,
    stop,
    supported: typeof window !== 'undefined' && 'vibrate' in navigator,
  };
};



