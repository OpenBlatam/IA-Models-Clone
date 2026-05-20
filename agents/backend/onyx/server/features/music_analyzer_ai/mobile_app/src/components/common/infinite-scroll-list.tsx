import React, { memo, useCallback, useMemo } from 'react';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { OptimizedFlatList } from './optimized-flatlist';
import { COLORS, SPACING } from '../../constants/config';
import type { ListRenderItem } from 'react-native';

interface InfiniteScrollListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  onLoadMore: () => void;
  hasMore: boolean;
  isLoadingMore: boolean;
  itemHeight?: number;
  estimatedItemHeight?: number;
  threshold?: number;
  loadingComponent?: React.ReactNode;
  endMessage?: string;
}

function InfiniteScrollListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  onLoadMore,
  hasMore,
  isLoadingMore,
  itemHeight,
  estimatedItemHeight,
  threshold = 0.5,
  loadingComponent,
  endMessage = 'No more items',
}: InfiniteScrollListProps<T>) {
  const handleEndReached = useCallback(() => {
    if (hasMore && !isLoadingMore) {
      onLoadMore();
    }
  }, [hasMore, isLoadingMore, onLoadMore]);

  const ListFooter = useMemo(() => {
    if (isLoadingMore) {
      return loadingComponent || (
        <View style={styles.footer}>
          <ActivityIndicator size="small" color={COLORS.primary} />
        </View>
      );
    }

    if (!hasMore && data.length > 0) {
      return (
        <View style={styles.endMessage}>
          {/* End message can be displayed here */}
        </View>
      );
    }

    return null;
  }, [isLoadingMore, hasMore, data.length, loadingComponent]);

  return (
    <OptimizedFlatList
      data={data}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      itemHeight={itemHeight}
      estimatedItemHeight={estimatedItemHeight}
      onEndReached={handleEndReached}
      onEndReachedThreshold={threshold}
      ListFooterComponent={ListFooter}
    />
  );
}

export const InfiniteScrollList = memo(InfiniteScrollListComponent) as <T>(
  props: InfiniteScrollListProps<T>
) => React.ReactElement;

const styles = StyleSheet.create({
  footer: {
    padding: SPACING.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  endMessage: {
    padding: SPACING.md,
    alignItems: 'center',
  },
});

