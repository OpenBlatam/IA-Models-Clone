import React, { memo, useMemo, useCallback, ComponentProps } from 'react';
import { View, ListRenderItem, ViewToken } from 'react-native';
import { OptimizedFlatList } from './optimized-flatlist';
import { LoadingSpinner } from './loading-spinner';

interface VirtualizedListProps<T> extends ComponentProps<typeof OptimizedFlatList<T>> {
  onViewableItemsChanged?: (items: T[]) => void;
  viewabilityConfig?: {
    itemVisiblePercentThreshold?: number;
    minimumViewTime?: number;
  };
  enablePagination?: boolean;
  onLoadMore?: () => void;
  hasMore?: boolean;
  loadingMore?: boolean;
  itemHeight: number;
  estimatedItemHeight?: number;
  overscan?: number;
}

function VirtualizedListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  itemHeight,
  estimatedItemHeight,
  overscan = 5,
  onViewableItemsChanged,
  viewabilityConfig = {
    itemVisiblePercentThreshold: 50,
    minimumViewTime: 100,
  },
  enablePagination = false,
  onLoadMore,
  hasMore = false,
  loadingMore = false,
  onEndReachedThreshold = 0.5,
  ...props
}: VirtualizedListProps<T>) {
  const viewabilityConfigCallbackPairs = useMemo(
    () => [
      {
        viewabilityConfig,
        onViewableItemsChanged: (info: {
          viewableItems: Array<ViewToken>;
          changed: Array<ViewToken>;
        }) => {
          if (onViewableItemsChanged) {
            const visibleItems = info.viewableItems
              .filter((item) => item.isViewable)
              .map((item) => item.item)
              .filter((item): item is T => item !== null && item !== undefined);
            onViewableItemsChanged(visibleItems);
          }
        },
      },
    ],
    [onViewableItemsChanged, viewabilityConfig]
  );

  const handleEndReached = useCallback(() => {
    if (enablePagination && hasMore && !loadingMore && onLoadMore) {
      onLoadMore();
    }
  }, [enablePagination, hasMore, loadingMore, onLoadMore]);

  const memoizedRenderItem = useCallback<ListRenderItem<T>>(
    (info) => renderItem(info),
    [renderItem]
  );

  const ListFooter = useMemo(() => {
    if (!loadingMore) return null;
    return (
      <View style={{ padding: 16, alignItems: 'center' }}>
        <LoadingSpinner size="small" />
      </View>
    );
  }, [loadingMore]);

  return (
    <OptimizedFlatList
      data={data}
      renderItem={memoizedRenderItem}
      keyExtractor={keyExtractor}
      itemHeight={itemHeight}
      estimatedItemHeight={estimatedItemHeight}
      viewabilityConfigCallbackPairs={viewabilityConfigCallbackPairs}
      onEndReached={handleEndReached}
      onEndReachedThreshold={onEndReachedThreshold}
      ListFooterComponent={ListFooter}
      {...props}
    />
  );
}

export const VirtualizedList = memo(VirtualizedListComponent) as <T>(
  props: VirtualizedListProps<T>
) => React.ReactElement;
