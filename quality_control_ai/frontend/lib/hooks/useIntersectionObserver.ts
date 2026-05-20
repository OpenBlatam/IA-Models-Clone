import { useEffect, useRef, useState, RefObject } from 'react';

interface UseIntersectionObserverOptions extends IntersectionObserverInit {
  triggerOnce?: boolean;
}

export const useIntersectionObserver = <T extends HTMLElement = HTMLDivElement>(
  options: UseIntersectionObserverOptions = {}
): [RefObject<T>, boolean] => {
  const { triggerOnce = false, ...observerOptions } = options;
  const elementRef = useRef<T>(null);
  const [isIntersecting, setIsIntersecting] = useState(false);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(([entry]) => {
      setIsIntersecting(entry.isIntersecting);

      if (triggerOnce && entry.isIntersecting) {
        observer.disconnect();
      }
    }, observerOptions);

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [triggerOnce, observerOptions.root, observerOptions.rootMargin, observerOptions.threshold]);

  return [elementRef, isIntersecting];
};

