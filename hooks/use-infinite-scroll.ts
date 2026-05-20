import { useEffect, useRef, useCallback } from "react";

interface UseInfiniteScrollProps {
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  isLoading: boolean;
}

export function useInfiniteScroll({
  onLoadMore,
  hasMore,
  isLoading,
}: UseInfiniteScrollProps) {
  const observer = useRef<IntersectionObserver>();
  const ref = useCallback(
    (node: HTMLDivElement | null) => {
      if (isLoading) return;
      if (observer.current) observer.current.disconnect();

      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMore) {
          onLoadMore();
        }
      });

      if (node) observer.current.observe(node);
    },
    [isLoading, hasMore, onLoadMore]
  );

  return { ref };
} 