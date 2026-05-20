import { useState, useEffect, RefObject } from 'react';

interface Rect {
  top: number;
  left: number;
  bottom: number;
  right: number;
  width: number;
  height: number;
}

export const useRect = <T extends HTMLElement = HTMLDivElement>(
  ref: RefObject<T>
): Rect | null => {
  const [rect, setRect] = useState<Rect | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    const updateRect = (): void => {
      const boundingRect = ref.current!.getBoundingClientRect();
      setRect({
        top: boundingRect.top,
        left: boundingRect.left,
        bottom: boundingRect.bottom,
        right: boundingRect.right,
        width: boundingRect.width,
        height: boundingRect.height,
      });
    };

    updateRect();

    const resizeObserver = new ResizeObserver(updateRect);
    resizeObserver.observe(ref.current);

    window.addEventListener('scroll', updateRect, true);
    window.addEventListener('resize', updateRect);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('scroll', updateRect, true);
      window.removeEventListener('resize', updateRect);
    };
  }, [ref]);

  return rect;
};

