import React, { memo, useMemo, useCallback } from 'react';
import { FlatList, FlatListProps, ListRenderItem, Platform } from 'react-native';

interface OptimizedFlatListProps<T> extends Omit<FlatListProps<T>, 'removeClippedSubviews' | 'maxToRenderPerBatch' | 'windowSize' | 'initialNumToRender' | 'getItemLayout' | 'updateCellsBatchingPeriod'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  itemHeight?: number;
  estimatedItemHeight?: number;
  enableOptimizations?: boolean;
}

function OptimizedFlatListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  itemHeight,
  estimatedItemHeight,
  enableOptimizations = true,
  ...props
}: OptimizedFlatListProps<T>) {
  const getItemLayout = useMemo(() => {
    if (!itemHeight) return undefined;
    return (_: any, index: number) => ({
      length: itemHeight,
      offset: itemHeight * index,
      index,
    });
  }, [itemHeight]);

  const optimizedProps = useMemo(() => {
    if (!enableOptimizations) return {};
    return {
      removeClippedSubviews: Platform.OS !== 'web',
      maxToRenderPerBatch: 10,
      windowSize: 10,
      initialNumToRender: 10,
      updateCellsBatchingPeriod: 50,
      ...(getItemLayout && { getItemLayout }),
      ...(estimatedItemHeight && { estimatedItemHeight }),
    };
  }, [enableOptimizations, getItemLayout, estimatedItemHeight]);

  return (
    <FlatList
      data={data}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      {...optimizedProps}
      {...props}
    />
  );
}

export const OptimizedFlatList = memo(OptimizedFlatListComponent) as <T>(
  props: OptimizedFlatListProps<T>
) => React.ReactElement;