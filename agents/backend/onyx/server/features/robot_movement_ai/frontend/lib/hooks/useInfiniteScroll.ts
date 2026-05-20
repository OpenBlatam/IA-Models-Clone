import { useEffect, useRef, useCallback, useState } from 'react';
import { useIntersectionObserver } from './useIntersectionObserver';

export interface UseInfiniteScrollOptions {
  threshold?: number;
  rootMargin?: string;
  enabled?: boolean;
}

/**
 * Hook for infinite scroll functionality
 */
export function useInfiniteScroll(
  onLoadMore: () => void | Promise<void>,
  options: UseInfiniteScrollOptions = {}
) {
  const { threshold = 0.1, rootMargin = '100px', enabled = true } = options;
  const [isLoading, setIsLoading] = useState(false);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  const handleLoadMore = useCallback(async () => {
    if (isLoading || !enabled) return;

    setIsLoading(true);
    try {
      await onLoadMore();
    } finally {
      setIsLoading(false);
    }
  }, [onLoadMore, isLoading, enabled]);

  const { isIntersecting } = useIntersectionObserver(loadMoreRef, {
    threshold,
    rootMargin,
  });

  useEffect(() => {
    if (isIntersecting && enabled && !isLoading) {
      handleLoadMore();
    }
  }, [isIntersecting, enabled, isLoading, handleLoadMore]);

  return {
    loadMoreRef,
    isLoading,
  };
}



