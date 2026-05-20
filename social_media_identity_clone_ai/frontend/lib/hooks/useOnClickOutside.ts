import { useEffect, useRef } from 'react';

export const useOnClickOutside = <T extends HTMLElement = HTMLElement>(
  handler: (event: MouseEvent | TouchEvent) => void,
  ...refs: Array<React.RefObject<T> | null>
): void => {
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent | TouchEvent): void => {
      const isOutside = refs.every((ref) => {
        if (!ref || !ref.current) {
          return true;
        }
        return !ref.current.contains(event.target as Node);
      });

      if (isOutside) {
        handler(event);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
    };
  }, [handler, refs]);
};



