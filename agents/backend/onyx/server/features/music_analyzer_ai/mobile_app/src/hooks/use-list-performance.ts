import { useCallback, useMemo, useRef } from 'react';
import type { ListRenderItem, ViewToken } from 'react-native';

interface UseListPerformanceOptions<T> {
  onItemVisible?: (item: T) => void;
  onItemHidden?: (item: T) => void;
  trackVisibility?: boolean;
}

export function useListPerformance<T>(options: UseListPerformanceOptions<T> = {}) {
  const { onItemVisible, onItemHidden, trackVisibility = false } = options;
  const visibleItemsRef = useRef<Set<string>>(new Set());

  const handleViewableItemsChanged = useCallback(
    (info: { viewableItems: Array<ViewToken>; changed: Array<ViewToken> }) => {
      if (!trackVisibility) return;

      info.changed.forEach((item) => {
        const key = item.key || String(item.index);
        const wasVisible = visibleItemsRef.current.has(key);
        const isNowVisible = item.isViewable;

        if (isNowVisible && !wasVisible) {
          visibleItemsRef.current.add(key);
          if (onItemVisible && item.item) {
            onItemVisible(item.item as T);
          }
        } else if (!isNowVisible && wasVisible) {
          visibleItemsRef.current.delete(key);
          if (onItemHidden && item.item) {
            onItemHidden(item.item as T);
          }
        }
      });
    },
    [trackVisibility, onItemVisible, onItemHidden]
  );

  const viewabilityConfig = useMemo(
    () => ({
      itemVisiblePercentThreshold: 50,
      minimumViewTime: 100,
      waitForInteraction: false,
    }),
    []
  );

  const viewabilityConfigCallbackPairs = useMemo(
    () => [
      {
        viewabilityConfig,
        onViewableItemsChanged: handleViewableItemsChanged,
      },
    ],
    [viewabilityConfig, handleViewableItemsChanged]
  );

  const createOptimizedRenderItem = useCallback(
    <T,>(renderItem: ListRenderItem<T>): ListRenderItem<T> => {
      return (info) => renderItem(info);
    },
    []
  );

  return {
    viewabilityConfigCallbackPairs,
    viewabilityConfig,
    createOptimizedRenderItem,
    visibleItemsCount: visibleItemsRef.current.size,
  };
}

