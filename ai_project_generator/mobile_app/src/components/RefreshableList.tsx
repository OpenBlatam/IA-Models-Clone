import React from 'react';
import { FlatList, FlatListProps, RefreshControl, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface RefreshableListProps<T> extends FlatListProps<T> {
  refreshing: boolean;
  onRefresh: () => void;
  refreshColors?: string[];
}

export function RefreshableList<T>({
  refreshing,
  onRefresh,
  refreshColors,
  ...props
}: RefreshableListProps<T>) {
  const { theme } = useTheme();

  const defaultColors = refreshColors || [theme.primary];

  return (
    <FlatList
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          colors={defaultColors}
          tintColor={theme.primary}
        />
      }
      {...props}
    />
  );
}

