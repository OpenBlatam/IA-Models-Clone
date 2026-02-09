import { useEffect, useRef } from 'react';

export const useClickOutside = (
  handler: () => void,
  enabled: boolean = true
) => {
  const ref = useRef<any>(null);

  useEffect(() => {
    if (!enabled) return;

    const handleClickOutside = (event: any) => {
      if (ref.current && !ref.current.contains(event.target)) {
        handler();
      }
    };

    // Note: In React Native, we use different event handling
    // This is a simplified version for web compatibility
    document?.addEventListener('mousedown', handleClickOutside);

    return () => {
      document?.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handler, enabled]);

  return ref;
};

