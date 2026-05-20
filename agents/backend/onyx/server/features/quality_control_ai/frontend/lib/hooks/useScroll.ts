import { useState, useEffect } from 'react';

interface ScrollPosition {
  x: number;
  y: number;
}

export const useScroll = (): ScrollPosition => {
  const [position, setPosition] = useState<ScrollPosition>({ x: 0, y: 0 });

  useEffect(() => {
    const updatePosition = (): void => {
      setPosition({
        x: window.scrollX || window.pageXOffset,
        y: window.scrollY || window.pageYOffset,
      });
    };

    window.addEventListener('scroll', updatePosition);
    updatePosition();

    return () => {
      window.removeEventListener('scroll', updatePosition);
    };
  }, []);

  return position;
};

export const useScrollDirection = (): 'up' | 'down' | null => {
  const [direction, setDirection] = useState<'up' | 'down' | null>(null);
  const [lastScrollY, setLastScrollY] = useState(0);

  useEffect(() => {
    const updateDirection = (): void => {
      const scrollY = window.scrollY;
      const direction = scrollY > lastScrollY ? 'down' : 'up';
      setDirection(direction);
      setLastScrollY(scrollY);
    };

    window.addEventListener('scroll', updateDirection);
    return () => {
      window.removeEventListener('scroll', updateDirection);
    };
  }, [lastScrollY]);

  return direction;
};

