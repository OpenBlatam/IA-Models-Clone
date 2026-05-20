import { useState, useRef, useEffect } from 'react';

export const useHover = <T extends HTMLElement = HTMLDivElement>(): [React.RefObject<T>, boolean] => {
  const [isHovered, setIsHovered] = useState(false);
  const ref = useRef<T>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    const handleMouseEnter = (): void => setIsHovered(true);
    const handleMouseLeave = (): void => setIsHovered(false);

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  return [ref, isHovered];
};



