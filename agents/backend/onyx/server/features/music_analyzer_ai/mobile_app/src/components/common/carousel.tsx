import React, { useRef, useState } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  NativeScrollEvent,
  NativeSyntheticEvent,
} from 'react-native';
import { COLORS, SPACING } from '../../constants/config';

interface CarouselProps<T> {
  data: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemWidth?: number;
  spacing?: number;
  onPageChange?: (index: number) => void;
  autoPlay?: boolean;
  autoPlayInterval?: number;
}

/**
 * Carousel component
 * Horizontal scrolling carousel
 */
export function Carousel<T>({
  data,
  renderItem,
  itemWidth,
  spacing = SPACING.md,
  onPageChange,
  autoPlay = false,
  autoPlayInterval = 3000,
}: CarouselProps<T>) {
  const flatListRef = useRef<FlatList>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = React.useRef<NodeJS.Timeout | null>(null);

  React.useEffect(() => {
    if (autoPlay && data.length > 1) {
      intervalRef.current = setInterval(() => {
        const nextIndex = (currentIndex + 1) % data.length;
        flatListRef.current?.scrollToIndex({
          index: nextIndex,
          animated: true,
        });
        setCurrentIndex(nextIndex);
        onPageChange?.(nextIndex);
      }, autoPlayInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoPlay, autoPlayInterval, currentIndex, data.length, onPageChange]);

  const handleScroll = (event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const scrollPosition = event.nativeEvent.contentOffset.x;
    const index = Math.round(scrollPosition / ((itemWidth || 300) + spacing));
    if (index !== currentIndex && index >= 0 && index < data.length) {
      setCurrentIndex(index);
      onPageChange?.(index);
    }
  };

  return (
    <View style={styles.container}>
      <FlatList
        ref={flatListRef}
        data={data}
        renderItem={({ item, index }) => (
          <View
            style={[
              styles.item,
              {
                width: itemWidth || 300,
                marginRight: index < data.length - 1 ? spacing : 0,
              },
            ]}
          >
            {renderItem(item, index)}
          </View>
        )}
        keyExtractor={(_, index) => `carousel-item-${index}`}
        horizontal
        showsHorizontalScrollIndicator={false}
        pagingEnabled={!itemWidth}
        snapToInterval={itemWidth ? itemWidth + spacing : undefined}
        decelerationRate="fast"
        onScroll={handleScroll}
        scrollEventThrottle={16}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
  item: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});

