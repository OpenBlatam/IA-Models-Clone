import React, { useState, useCallback } from 'react';
import { View, ScrollView, StyleSheet, Dimensions } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface MasonryListProps<T> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  numColumns?: number;
  spacing?: number;
  keyExtractor?: (item: T, index: number) => string;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export function MasonryList<T>({
  data,
  renderItem,
  numColumns = 2,
  spacing: itemSpacing = spacing.md,
  keyExtractor = (item, index) => index.toString(),
}: MasonryListProps<T>) {
  const { theme } = useTheme();
  const [columnHeights, setColumnHeights] = useState<number[]>(
    new Array(numColumns).fill(0)
  );

  const itemWidth = (SCREEN_WIDTH - itemSpacing * (numColumns + 1)) / numColumns;

  const getShortestColumn = useCallback(() => {
    return columnHeights.indexOf(Math.min(...columnHeights));
  }, [columnHeights]);

  const columns: T[][] = Array.from({ length: numColumns }, () => []);

  data.forEach((item, index) => {
    const columnIndex = getShortestColumn();
    columns[columnIndex].push(item);
    setColumnHeights((prev) => {
      const newHeights = [...prev];
      newHeights[columnIndex] += 200; // Approximate height
      return newHeights;
    });
  });

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: theme.background }]}
      contentContainerStyle={styles.content}
    >
      <View style={styles.columns}>
        {columns.map((column, columnIndex) => (
          <View
            key={columnIndex}
            style={[
              styles.column,
              {
                width: itemWidth,
                marginRight: columnIndex < numColumns - 1 ? itemSpacing : 0,
              },
            ]}
          >
            {column.map((item, itemIndex) => (
              <View
                key={keyExtractor(item, itemIndex)}
                style={[styles.item, { marginBottom: itemSpacing }]}
              >
                {renderItem(item, itemIndex)}
              </View>
            ))}
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: spacing.md,
  },
  columns: {
    flexDirection: 'row',
  },
  column: {
    flex: 1,
  },
  item: {
    // Item styles handled inline
  },
});

