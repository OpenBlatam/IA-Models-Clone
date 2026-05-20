import React, { useMemo, useCallback } from 'react';
import { FlatList, FlatListProps, ListRenderItem } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { useOptimizedFlatList } from '../hooks/useOptimizedFlatList';

interface VirtualizedListProps<T> extends Omit<FlatListProps<T>, 'data' | 'renderItem'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  estimatedItemSize?: number;
  emptyComponent?: React.ReactNode;
  loadingComponent?: React.ReactNode;
  isLoading?: boolean;
}

function VirtualizedList<T>({
  data,
  renderItem,
  estimatedItemSize = 50,
  emptyComponent,
  loadingComponent,
  isLoading,
  ...props
}: VirtualizedListProps<T>) {
  const { colors } = useTheme();

  const optimizedProps = useOptimizedFlatList<T>({
    itemHeight: estimatedItemSize,
    estimatedItemSize,
  });

  const getItemLayout = useCallback(
    (_: unknown, index: number) => ({
      length: estimatedItemSize,
      offset: estimatedItemSize * index,
      index,
    }),
    [estimatedItemSize]
  );

  const listEmptyComponent = useMemo(() => {
    if (isLoading && loadingComponent) {
      return loadingComponent;
    }
    if (data.length === 0 && emptyComponent) {
      return emptyComponent;
    }
    return null;
  }, [isLoading, loadingComponent, data.length, emptyComponent]);

  return (
    <FlatList
      data={data}
      renderItem={renderItem}
      getItemLayout={getItemLayout}
      ListEmptyComponent={listEmptyComponent}
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      windowSize={10}
      initialNumToRender={10}
      {...optimizedProps}
      {...props}
    />
  );
}

export default React.memo(VirtualizedList) as <T>(
  props: VirtualizedListProps<T>
) => React.ReactElement;

