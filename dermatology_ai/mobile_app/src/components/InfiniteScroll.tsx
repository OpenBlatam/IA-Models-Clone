import React, { useCallback, useMemo } from 'react';
import { FlatList, FlatListProps, ListRenderItem, View } from 'react-native';
import { useInfiniteScroll } from '../hooks/useInfiniteScroll';
import { useOptimizedFlatList } from '../hooks/useOptimizedFlatList';

interface InfiniteScrollProps<T> extends Omit<FlatListProps<T>, 'onEndReached' | 'data' | 'renderItem'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  onLoadMore: () => Promise<void> | void;
  hasMore: boolean;
  threshold?: number;
  loadingComponent?: React.ReactNode;
  itemHeight?: number;
}

function InfiniteScroll<T>({
  data,
  renderItem,
  onLoadMore,
  hasMore,
  threshold = 0.5,
  loadingComponent,
  itemHeight,
  ...props
}: InfiniteScrollProps<T>) {
  const { lastElementRef, isLoading } = useInfiniteScroll({
    threshold,
    onLoadMore,
    hasMore,
  });

  const optimizedProps = useOptimizedFlatList<T>({
    itemHeight,
    estimatedItemSize: itemHeight || 50,
  });

  const renderItemWithRef = useCallback(
    (itemInfo: { item: T; index: number }) => {
      const isLast = itemInfo.index === data.length - 1;
      return (
        <View ref={isLast ? lastElementRef : null}>
          {renderItem(itemInfo)}
        </View>
      );
    },
    [data.length, lastElementRef, renderItem]
  );

  const footerComponent = useMemo(() => {
    if (isLoading && hasMore) {
      return loadingComponent;
    }
    return null;
  }, [isLoading, hasMore, loadingComponent]);

  return (
    <FlatList
      data={data}
      renderItem={renderItemWithRef}
      ListFooterComponent={footerComponent}
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

export default React.memo(InfiniteScroll) as <T>(
  props: InfiniteScrollProps<T>
) => React.ReactElement;

