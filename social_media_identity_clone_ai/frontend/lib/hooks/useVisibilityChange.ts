import { useEffect, useState } from 'react';

export const useVisibilityChange = (): boolean => {
  const [isVisible, setIsVisible] = useState(() => {
    if (typeof document === 'undefined') {
      return true;
    }
    return !document.hidden;
  });

  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }

    const handleVisibilityChange = (): void => {
      setIsVisible(!document.hidden);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  return isVisible;
};



