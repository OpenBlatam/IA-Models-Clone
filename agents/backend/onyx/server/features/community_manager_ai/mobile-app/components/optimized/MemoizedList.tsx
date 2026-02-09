import { ReactNode, useMemo } from 'react';
import { FlatList, FlatListProps, ListRenderItem } from 'react-native';

interface MemoizedListProps<T> extends Omit<FlatListProps<T>, 'renderItem'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
}

export function MemoizedList<T>({
  data,
  renderItem,
  keyExtractor,
  ...props
}: MemoizedListProps<T>) {
  const memoizedData = useMemo(() => data, [data]);
  const memoizedKeyExtractor = useMemo(() => keyExtractor, [keyExtractor]);
  const memoizedRenderItem = useMemo(() => renderItem, [renderItem]);

  return (
    <FlatList
      data={memoizedData}
      renderItem={memoizedRenderItem}
      keyExtractor={memoizedKeyExtractor}
      removeClippedSubviews
      maxToRenderPerBatch={10}
      updateCellsBatchingPeriod={50}
      initialNumToRender={10}
      windowSize={10}
      {...props}
    />
  );
}


