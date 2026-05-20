'use client';

import { useEffect, useRef, useState } from 'react';
import { useIntersectionObserver } from './useIntersectionObserver';

interface UseInfiniteScrollOptions {
  hasMore: boolean;
  isLoading?: boolean;
  onLoadMore: () => void | Promise<void>;
  threshold?: number;
}

export function useInfiniteScroll({
  hasMore,
  isLoading = false,
  onLoadMore,
  threshold = 0.1,
}: UseInfiniteScrollOptions) {
  const [triggerRef, isIntersecting] = useIntersectionObserver({
    threshold,
    triggerOnce: false,
  });

  const [isLoadingMore, setIsLoadingMore] = useState(false);

  useEffect(() => {
    if (isIntersecting && hasMore && !isLoading && !isLoadingMore) {
      setIsLoadingMore(true);
      Promise.resolve(onLoadMore()).finally(() => {
        setIsLoadingMore(false);
      });
    }
  }, [isIntersecting, hasMore, isLoading, isLoadingMore, onLoadMore]);

  return {
    triggerRef,
    isLoadingMore,
  };
}

