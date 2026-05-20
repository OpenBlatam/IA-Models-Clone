import React, { useMemo } from 'react';
import { VirtualizedList as RNVirtualizedList, VirtualizedListProps } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';

interface CustomVirtualizedListProps<T> extends Omit<VirtualizedListProps<T>, 'renderItem'> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor?: (item: T, index: number) => string;
}

export function VirtualizedList<T>({
  data,
  renderItem,
  keyExtractor,
  ...props
}: CustomVirtualizedListProps<T>) {
  const { theme } = useTheme();

  const getItem = (data: T[], index: number) => data[index];
  const getItemCount = () => data.length;

  const memoizedRenderItem = useMemo(
    () => ({ item, index }: { item: T; index: number }) => renderItem(item, index),
    [renderItem]
  );

  return (
    <RNVirtualizedList
      data={data}
      renderItem={memoizedRenderItem}
      getItem={getItem}
      getItemCount={getItemCount}
      keyExtractor={keyExtractor || ((item, index) => index.toString())}
      {...props}
    />
  );
}

