import { useState, useCallback, useRef } from 'react';

interface UseInfiniteScrollOptions {
  threshold?: number;
  onLoadMore: () => Promise<void> | void;
  hasMore: boolean;
}

export const useInfiniteScroll = ({
  threshold = 0.5,
  onLoadMore,
  hasMore,
}: UseInfiniteScrollOptions) => {
  const [isLoading, setIsLoading] = useState(false);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const lastElementRef = useCallback(
    (node: any) => {
      if (isLoading) return;
      if (observerRef.current) observerRef.current.disconnect();

      observerRef.current = new IntersectionObserver(
        (entries) => {
          if (entries[0].isIntersecting && hasMore) {
            setIsLoading(true);
            Promise.resolve(onLoadMore()).finally(() => {
              setIsLoading(false);
            });
          }
        },
        { threshold }
      );

      if (node) observerRef.current.observe(node);
    },
    [isLoading, hasMore, onLoadMore, threshold]
  );

  return { lastElementRef, isLoading };
};

