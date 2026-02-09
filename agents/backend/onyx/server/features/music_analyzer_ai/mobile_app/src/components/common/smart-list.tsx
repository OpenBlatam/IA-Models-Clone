import React, { memo, useMemo, useCallback } from 'react';
import { View, StyleSheet, ActivityIndicator } from 'react-native';
import { OptimizedFlatList } from './optimized-flatlist';
import { VirtualizedList } from './virtualized-list';
import { EmptyState } from './empty-state';
import { ErrorState } from './error-state';
import { SkeletonScreen } from './skeleton-screen';
import { COLORS, SPACING } from '../../constants/config';
import type { ListRenderItem } from 'react-native';

interface SmartListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  isLoading?: boolean;
  error?: unknown;
  onRetry?: () => void;
  emptyMessage?: string;
  emptyIcon?: string;
  itemHeight?: number;
  estimatedItemHeight?: number;
  enablePagination?: boolean;
  onLoadMore?: () => void;
  hasMore?: boolean;
  loadingMore?: boolean;
  showSkeleton?: boolean;
  skeletonItemCount?: number;
}

function SmartListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  isLoading = false,
  error,
  onRetry,
  emptyMessage = 'No items found',
  emptyIcon = '📭',
  itemHeight,
  estimatedItemHeight,
  enablePagination = false,
  onLoadMore,
  hasMore = false,
  loadingMore = false,
  showSkeleton = true,
  skeletonItemCount = 5,
}: SmartListProps<T>) {
  const ListComponent = useMemo(() => {
    if (itemHeight) {
      return VirtualizedList;
    }
    return OptimizedFlatList;
  }, [itemHeight]);

  const ListFooter = useMemo(() => {
    if (!loadingMore) return null;
    return (
      <View style={styles.footer}>
        <ActivityIndicator size="small" color={COLORS.primary} />
      </View>
    );
  }, [loadingMore]);

  const handleEndReached = useCallback(() => {
    if (enablePagination && hasMore && !loadingMore && onLoadMore) {
      onLoadMore();
    }
  }, [enablePagination, hasMore, loadingMore, onLoadMore]);

  if (isLoading && showSkeleton) {
    return <SkeletonScreen itemCount={skeletonItemCount} />;
  }

  if (error) {
    return (
      <ErrorState
        error={error}
        onRetry={onRetry}
        showDetails={__DEV__}
      />
    );
  }

  if (!data || data.length === 0) {
    return (
      <EmptyState
        icon={emptyIcon}
        title={emptyMessage}
        message="Try adjusting your search or filters"
      />
    );
  }

  const listProps = itemHeight
    ? {
        itemHeight,
        estimatedItemHeight,
        enablePagination,
        onLoadMore: handleEndReached,
        hasMore,
        loadingMore,
      }
    : {
        estimatedItemHeight,
      };

  return (
    <ListComponent
      data={data}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      ListFooterComponent={ListFooter}
      onEndReached={handleEndReached}
      onEndReachedThreshold={0.5}
      {...listProps}
    />
  );
}

export const SmartList = memo(SmartListComponent) as <T>(
  props: SmartListProps<T>
) => React.ReactElement;

const styles = StyleSheet.create({
  footer: {
    padding: SPACING.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

