import { ReactNode, useMemo } from 'react';
import { FlatList, FlatListProps, ListRenderItem } from 'react-native';

interface VirtualizedListProps<T> extends Omit<FlatListProps<T>, 'renderItem'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  estimatedItemSize?: number;
}

export function VirtualizedList<T>({
  data,
  renderItem,
  keyExtractor,
  estimatedItemSize = 100,
  ...props
}: VirtualizedListProps<T>) {
  const getItemLayout = useMemo(
    () =>
      (data: ArrayLike<T> | null | undefined, index: number) => ({
        length: estimatedItemSize,
        offset: estimatedItemSize * index,
        index,
      }),
    [estimatedItemSize]
  );

  return (
    <FlatList
      data={data}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      getItemLayout={getItemLayout}
      removeClippedSubviews
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      initialNumToRender={10}
      windowSize={10}
      {...props}
    />
  );
}


