import { useEffect } from 'react';

interface UseBeforeUnloadOptions {
  enabled?: boolean;
  message?: string;
}

export const useBeforeUnload = (options: UseBeforeUnloadOptions = {}): void => {
  const { enabled = true, message } = options;

  useEffect(() => {
    if (!enabled) return;

    const handleBeforeUnload = (event: BeforeUnloadEvent): void => {
      if (message) {
        event.preventDefault();
        event.returnValue = message;
        return message;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [enabled, message]);
};

