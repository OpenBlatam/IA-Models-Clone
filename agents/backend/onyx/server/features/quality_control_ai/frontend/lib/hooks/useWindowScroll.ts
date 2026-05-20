import { useEffect, useState } from 'react';

interface ScrollPosition {
  x: number;
  y: number;
}

export const useWindowScroll = (): ScrollPosition => {
  const [position, setPosition] = useState<ScrollPosition>({
    x: typeof window !== 'undefined' ? window.scrollX : 0,
    y: typeof window !== 'undefined' ? window.scrollY : 0,
  });

  useEffect(() => {
    const handleScroll = (): void => {
      setPosition({
        x: window.scrollX,
        y: window.scrollY,
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return position;
};

export const useScrollToTop = (): (() => void) => {
  return () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
};

export const useScrollToBottom = (): (() => void) => {
  return () => {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' });
  };
};

export const useScrollTo = (): ((options: ScrollToOptions) => void) => {
  return (options: ScrollToOptions) => {
    window.scrollTo(options);
  };
};

