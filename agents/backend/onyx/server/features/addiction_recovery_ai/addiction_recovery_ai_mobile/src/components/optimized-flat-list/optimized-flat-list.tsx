import React, { memo, useMemo, useCallback } from 'react';
import { FlashList, FlashListProps } from '@shopify/flash-list';
import type { ListRenderItem } from '@shopify/flash-list';

interface OptimizedFlatListProps<T> extends Omit<FlashListProps<T>, 'renderItem'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  estimatedItemSize: number;
  keyExtractor?: (item: T, index: number) => string;
}

function OptimizedFlatListComponent<T>({
  data,
  renderItem,
  estimatedItemSize,
  keyExtractor,
  ...props
}: OptimizedFlatListProps<T>): JSX.Element {
  const memoizedKeyExtractor = useCallback(
    (item: T, index: number) => {
      if (keyExtractor) {
        return keyExtractor(item, index);
      }
      if (typeof item === 'object' && item !== null && 'id' in item) {
        return String((item as { id: unknown }).id);
      }
      return `item-${index}`;
    },
    [keyExtractor]
  );

  const memoizedRenderItem = useCallback(
    (info: Parameters<ListRenderItem<T>>[0]) => renderItem(info),
    [renderItem]
  );

  const memoizedData = useMemo(() => data, [data]);

  return (
    <FlashList
      data={memoizedData}
      renderItem={memoizedRenderItem}
      keyExtractor={memoizedKeyExtractor}
      estimatedItemSize={estimatedItemSize}
      removeClippedSubviews
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      initialNumToRender={10}
      windowSize={10}
      {...props}
    />
  );
}

export const OptimizedFlatList = memo(OptimizedFlatListComponent) as <T>(
  props: OptimizedFlatListProps<T>
) => JSX.Element;

