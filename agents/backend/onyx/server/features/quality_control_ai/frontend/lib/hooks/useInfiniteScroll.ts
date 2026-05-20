import { useEffect, useRef, useCallback } from 'react';

interface UseInfiniteScrollOptions {
  hasMore: boolean;
  isLoading: boolean;
  onLoadMore: () => void;
  threshold?: number;
  root?: Element | null;
  rootMargin?: string;
}

export const useInfiniteScroll = (options: UseInfiniteScrollOptions) => {
  const {
    hasMore,
    isLoading,
    onLoadMore,
    threshold = 100,
    root = null,
    rootMargin = '0px',
  } = options;

  const observerRef = useRef<IntersectionObserver | null>(null);
  const elementRef = useRef<HTMLElement | null>(null);

  const handleObserver = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      const [entry] = entries;
      if (entry.isIntersecting && hasMore && !isLoading) {
        onLoadMore();
      }
    },
    [hasMore, isLoading, onLoadMore]
  );

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    observerRef.current = new IntersectionObserver(handleObserver, {
      root,
      rootMargin,
      threshold: 0.1,
    });

    observerRef.current.observe(element);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [handleObserver, root, rootMargin]);

  const setElementRef = useCallback((node: HTMLElement | null) => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    elementRef.current = node;

    if (node) {
      observerRef.current = new IntersectionObserver(handleObserver, {
        root,
        rootMargin,
        threshold: 0.1,
      });
      observerRef.current.observe(node);
    }
  }, [handleObserver, root, rootMargin]);

  return { ref: setElementRef };
};

