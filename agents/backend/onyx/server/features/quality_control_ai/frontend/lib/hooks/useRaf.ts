import { useEffect, useRef } from 'react';

export const useRaf = (callback: () => void, isActive = true): void => {
  const rafRef = useRef<number>();

  useEffect(() => {
    if (!isActive) return;

    const animate = (): void => {
      callback();
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [callback, isActive]);
};

