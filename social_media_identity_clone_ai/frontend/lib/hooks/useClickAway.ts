import { useEffect, useRef } from 'react';

export const useClickAway = <T extends HTMLElement = HTMLElement>(
  handler: (event: MouseEvent | TouchEvent) => void
): React.RefObject<T> => {
  const ref = useRef<T>(null);

  useEffect(() => {
    const handleClickAway = (event: MouseEvent | TouchEvent): void => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        handler(event);
      }
    };

    document.addEventListener('mousedown', handleClickAway);
    document.addEventListener('touchstart', handleClickAway);

    return () => {
      document.removeEventListener('mousedown', handleClickAway);
      document.removeEventListener('touchstart', handleClickAway);
    };
  }, [handler]);

  return ref;
};



