import { useCallback, useState } from 'react';

export const useVibrate = () => {
  const [isSupported, setIsSupported] = useState(
    typeof navigator !== 'undefined' && 'vibrate' in navigator
  );

  const vibrate = useCallback(
    (pattern: number | number[]): boolean => {
      if (!isSupported) {
        return false;
      }

      try {
        navigator.vibrate(pattern);
        return true;
      } catch {
        return false;
      }
    },
    [isSupported]
  );

  const stop = useCallback((): boolean => {
    if (!isSupported) {
      return false;
    }

    try {
      navigator.vibrate(0);
      return true;
    } catch {
      return false;
    }
  }, [isSupported]);

  return { vibrate, stop, isSupported };
};

