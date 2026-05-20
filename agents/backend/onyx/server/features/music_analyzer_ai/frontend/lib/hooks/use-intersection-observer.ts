/**
 * Custom hook for Intersection Observer API.
 * Provides element visibility detection for lazy loading and animations.
 */

import { useEffect, useRef, useState, type RefObject } from 'react';

/**
 * Options for Intersection Observer.
 */
export interface UseIntersectionObserverOptions {
  threshold?: number | number[];
  root?: Element | null;
  rootMargin?: string;
  triggerOnce?: boolean;
}

/**
 * Return type for useIntersectionObserver hook.
 */
export interface UseIntersectionObserverReturn {
  ref: RefObject<Element>;
  isIntersecting: boolean;
  entry: IntersectionObserverEntry | null;
}

/**
 * Custom hook for Intersection Observer.
 * Detects when an element enters or leaves the viewport.
 *
 * @param options - Intersection Observer options
 * @returns Intersection observer state and ref
 */
export function useIntersectionObserver(
  options: UseIntersectionObserverOptions = {}
): UseIntersectionObserverReturn {
  const {
    threshold = 0,
    root = null,
    rootMargin = '0px',
    triggerOnce = false,
  } = options;

  const [isIntersecting, setIsIntersecting] = useState(false);
  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null);
  const elementRef = useRef<Element>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        setIsIntersecting(entry.isIntersecting);
        setEntry(entry);

        if (entry.isIntersecting && triggerOnce) {
          observer.disconnect();
        }
      },
      {
        threshold,
        root,
        rootMargin,
      }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [threshold, root, rootMargin, triggerOnce]);

  return {
    ref: elementRef as RefObject<Element>,
    isIntersecting,
    entry,
  };
}

