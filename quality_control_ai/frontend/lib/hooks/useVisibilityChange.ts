import { useEffect, useState } from 'react';

export const useVisibilityChange = (): boolean => {
  const [isVisible, setIsVisible] = useState(
    typeof document !== 'undefined' ? !document.hidden : true
  );

  useEffect(() => {
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

