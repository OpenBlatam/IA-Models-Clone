import { useState, useEffect, useRef } from 'react';

interface ScrollPosition {
  x: number;
  y: number;
}

export const useScroll = (): ScrollPosition => {
  const [scrollPosition, setScrollPosition] = useState<ScrollPosition>(() => {
    if (typeof window === 'undefined') {
      return { x: 0, y: 0 };
    }
    return {
      x: window.scrollX,
      y: window.scrollY,
    };
  });

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleScroll = () => {
      setScrollPosition({
        x: window.scrollX,
        y: window.scrollY,
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return scrollPosition;
};

export const useScrollDirection = (): 'up' | 'down' | null => {
  const [scrollDirection, setScrollDirection] = useState<'up' | 'down' | null>(null);
  const prevScrollY = useRef<number>(0);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      if (currentScrollY > prevScrollY.current) {
        setScrollDirection('down');
      } else if (currentScrollY < prevScrollY.current) {
        setScrollDirection('up');
      }
      prevScrollY.current = currentScrollY;
    };

    prevScrollY.current = window.scrollY;
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return scrollDirection;
};

