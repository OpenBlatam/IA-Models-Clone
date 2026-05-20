import React, { memo, useCallback } from 'react';
import { RefreshControl } from 'react-native';
import { OptimizedFlatList } from './optimized-flatlist';
import { COLORS } from '../../constants/config';
import type { ListRenderItem } from 'react-native';

interface RefreshableListProps<T> {
  data: T[];
  renderItem: ListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  refreshing: boolean;
  onRefresh: () => void;
  refreshColors?: string[];
  itemHeight?: number;
  estimatedItemHeight?: number;
}

function RefreshableListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  refreshing,
  onRefresh,
  refreshColors = [COLORS.primary],
  itemHeight,
  estimatedItemHeight,
  ...props
}: RefreshableListProps<T>) {
  const refreshControl = useCallback(
    () => (
      <RefreshControl
        refreshing={refreshing}
        onRefresh={onRefresh}
        colors={refreshColors}
        tintColor={refreshColors[0]}
      />
    ),
    [refreshing, onRefresh, refreshColors]
  );

  return (
    <OptimizedFlatList
      data={data}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      itemHeight={itemHeight}
      estimatedItemHeight={estimatedItemHeight}
      refreshControl={refreshControl()}
      {...props}
    />
  );
}

export const RefreshableList = memo(RefreshableListComponent) as <T>(
  props: RefreshableListProps<T>
) => React.ReactElement;

