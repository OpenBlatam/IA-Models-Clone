import { useEffect, useRef } from 'react';
import { trapFocus } from '@/lib/utils/accessibility';

export const useFocusTrap = (isActive: boolean) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) {
      return;
    }

    const cleanup = trapFocus(containerRef.current);
    return cleanup;
  }, [isActive]);

  return containerRef;
};



