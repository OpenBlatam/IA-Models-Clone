import { useRef, useCallback } from 'react';

interface UseDoubleClickOptions {
  onSingleClick?: () => void;
  onDoubleClick?: () => void;
  delay?: number;
}

export const useDoubleClick = (options: UseDoubleClickOptions = {}) => {
  const { onSingleClick, onDoubleClick, delay = 300 } = options;
  const clickCountRef = useRef(0);
  const timeoutRef = useRef<NodeJS.Timeout>();

  const handleClick = useCallback(() => {
    clickCountRef.current += 1;

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(() => {
      if (clickCountRef.current === 1) {
        onSingleClick?.();
      } else if (clickCountRef.current === 2) {
        onDoubleClick?.();
      }
      clickCountRef.current = 0;
    }, delay);
  }, [onSingleClick, onDoubleClick, delay]);

  return handleClick;
};

