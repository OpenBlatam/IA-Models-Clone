import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface WaterfallListProps<T> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  numColumns?: number;
  columnSpacing?: number;
  itemSpacing?: number;
  onEndReached?: () => void;
  onEndReachedThreshold?: number;
}

function WaterfallList<T>({
  data,
  renderItem,
  numColumns = 2,
  columnSpacing = 8,
  itemSpacing = 8,
  onEndReached,
  onEndReachedThreshold = 0.5,
}: WaterfallListProps<T>) {
  const { colors } = useTheme();
  const [columns, setColumns] = useState<Array<Array<{ item: T; index: number }>>>([]);
  const [columnHeights, setColumnHeights] = useState<number[]>([]);
  const screenWidth = Dimensions.get('window').width;
  const itemWidth = (screenWidth - columnSpacing * (numColumns + 1)) / numColumns;

  useEffect(() => {
    const newColumns: Array<Array<{ item: T; index: number }>> = Array(numColumns)
      .fill(null)
      .map(() => []);
    const newHeights = Array(numColumns).fill(0);

    data.forEach((item, index) => {
      const shortestColumnIndex = newHeights.indexOf(Math.min(...newHeights));
      newColumns[shortestColumnIndex].push({ item, index });
      newHeights[shortestColumnIndex] += 1;
    });

    setColumns(newColumns);
    setColumnHeights(newHeights);
  }, [data, numColumns]);

  const handleScroll = (event: any) => {
    const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
    const paddingToBottom = 20;
    const isCloseToBottom =
      layoutMeasurement.height + contentOffset.y >=
      contentSize.height - paddingToBottom;

    if (isCloseToBottom && onEndReached) {
      onEndReached();
    }
  };

  return (
    <ScrollView
      style={styles.container}
      onScroll={handleScroll}
      scrollEventThrottle={16}
      showsVerticalScrollIndicator={false}
    >
      <View
        style={[
          styles.columnsContainer,
          {
            paddingHorizontal: columnSpacing,
          },
        ]}
      >
        {columns.map((column, columnIndex) => (
          <View
            key={columnIndex}
            style={[
              styles.column,
              {
                width: itemWidth,
                marginRight: columnIndex < numColumns - 1 ? columnSpacing : 0,
              },
            ]}
          >
            {column.map(({ item, index }) => (
              <View
                key={index}
                style={[
                  styles.item,
                  {
                    marginBottom: itemSpacing,
                  },
                ]}
              >
                {renderItem(item, index)}
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
  columnsContainer: {
    flexDirection: 'row',
    paddingTop: 8,
  },
  column: {
    flex: 1,
  },
  item: {
    width: '100%',
  },
});

export default WaterfallList;

