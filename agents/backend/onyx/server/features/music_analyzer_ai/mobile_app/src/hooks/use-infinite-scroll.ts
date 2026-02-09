import { useState, useCallback, useRef } from 'react';
import { NativeScrollEvent, NativeSyntheticEvent } from 'react-native';

interface UseInfiniteScrollOptions {
  threshold?: number;
  onLoadMore: () => Promise<void> | void;
  hasMore: boolean;
  isLoading: boolean;
}

/**
 * Hook for infinite scroll functionality
 * Automatically loads more when near bottom
 */
export function useInfiniteScroll({
  threshold = 0.5,
  onLoadMore,
  hasMore,
  isLoading,
}: UseInfiniteScrollOptions) {
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const loadingRef = useRef(false);

  const handleScroll = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      const { layoutMeasurement, contentOffset, contentSize } =
        event.nativeEvent;

      const isNearBottom =
        layoutMeasurement.height + contentOffset.y >=
        contentSize.height * threshold;

      if (
        isNearBottom &&
        hasMore &&
        !isLoading &&
        !isLoadingMore &&
        !loadingRef.current
      ) {
        loadingRef.current = true;
        setIsLoadingMore(true);

        Promise.resolve(onLoadMore())
          .finally(() => {
            setIsLoadingMore(false);
            loadingRef.current = false;
          });
      }
    },
    [threshold, onLoadMore, hasMore, isLoading, isLoadingMore]
  );

  return {
    handleScroll,
    isLoadingMore,
  };
}

