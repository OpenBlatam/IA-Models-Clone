import React from 'react';
import { View, FlatList, StyleSheet, Dimensions } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface GridListProps<T> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  numColumns?: number;
  spacing?: number;
  keyExtractor?: (item: T, index: number) => string;
  onEndReached?: () => void;
  onEndReachedThreshold?: number;
  ListHeaderComponent?: React.ReactNode;
  ListFooterComponent?: React.ReactNode;
  ListEmptyComponent?: React.ReactNode;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export function GridList<T>({
  data,
  renderItem,
  numColumns = 2,
  spacing: itemSpacing = spacing.md,
  keyExtractor = (item, index) => index.toString(),
  onEndReached,
  onEndReachedThreshold = 0.5,
  ListHeaderComponent,
  ListFooterComponent,
  ListEmptyComponent,
}: GridListProps<T>) {
  const { theme } = useTheme();

  const itemWidth = (SCREEN_WIDTH - itemSpacing * (numColumns + 1)) / numColumns;

  return (
    <FlatList
      data={data}
      renderItem={({ item, index }) => (
        <View
          style={[
            styles.item,
            {
              width: itemWidth,
              marginRight: (index + 1) % numColumns !== 0 ? itemSpacing : 0,
              marginBottom: itemSpacing,
            },
          ]}
        >
          {renderItem(item, index)}
        </View>
      )}
      keyExtractor={keyExtractor}
      numColumns={numColumns}
      onEndReached={onEndReached}
      onEndReachedThreshold={onEndReachedThreshold}
      ListHeaderComponent={ListHeaderComponent}
      ListFooterComponent={ListFooterComponent}
      ListEmptyComponent={ListEmptyComponent}
      contentContainerStyle={styles.content}
      columnWrapperStyle={numColumns > 1 ? styles.row : undefined}
    />
  );
}

const styles = StyleSheet.create({
  content: {
    padding: spacing.md,
  },
  row: {
    flexDirection: 'row',
  },
  item: {
    // Item styles handled inline
  },
});

