import React, { useEffect, useCallback } from 'react';
import { FlatList, FlatListProps, ActivityIndicator, View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, typography } from '../theme/colors';

interface InfiniteScrollProps<T> extends Omit<FlatListProps<T>, 'onEndReached' | 'onEndReachedThreshold'> {
  data: T[];
  renderItem: FlatListProps<T>['renderItem'];
  onLoadMore?: () => void;
  hasMore?: boolean;
  loading?: boolean;
  loadingComponent?: React.ReactNode;
  emptyComponent?: React.ReactNode;
  errorComponent?: React.ReactNode;
  error?: string | null;
}

export function InfiniteScroll<T>({
  data,
  renderItem,
  onLoadMore,
  hasMore = true,
  loading = false,
  loadingComponent,
  emptyComponent,
  errorComponent,
  error,
  ...flatListProps
}: InfiniteScrollProps<T>) {
  const { theme } = useTheme();

  const handleEndReached = useCallback(() => {
    if (hasMore && !loading && onLoadMore) {
      onLoadMore();
    }
  }, [hasMore, loading, onLoadMore]);

  const defaultLoadingComponent = (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="small" color={theme.primary} />
      <Text style={[styles.loadingText, { color: theme.textSecondary }]}>
        Cargando más...
      </Text>
    </View>
  );

  const defaultEmptyComponent = (
    <View style={styles.emptyContainer}>
      <Text style={[styles.emptyText, { color: theme.textSecondary }]}>
        No hay elementos para mostrar
      </Text>
    </View>
  );

  const defaultErrorComponent = error ? (
    <View style={styles.errorContainer}>
      <Text style={[styles.errorText, { color: theme.error }]}>
        {error}
      </Text>
    </View>
  ) : null;

  return (
    <FlatList
      data={data}
      renderItem={renderItem}
      onEndReached={handleEndReached}
      onEndReachedThreshold={0.5}
      ListFooterComponent={
        loading && hasMore
          ? loadingComponent || defaultLoadingComponent
          : null
      }
      ListEmptyComponent={
        !loading
          ? emptyComponent || defaultEmptyComponent
          : null
      }
      ListHeaderComponent={
        error
          ? errorComponent || defaultErrorComponent
          : null
      }
      {...flatListProps}
    />
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    padding: spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: spacing.sm,
  },
  loadingText: {
    ...typography.bodySmall,
  },
  emptyContainer: {
    padding: spacing.xl,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyText: {
    ...typography.body,
  },
  errorContainer: {
    padding: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorText: {
    ...typography.body,
  },
});

