import React, { useRef, useState } from 'react';
import { View, ScrollView, StyleSheet, Dimensions, NativeScrollEvent, NativeSyntheticEvent } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing } from '../theme/colors';

interface CarouselProps {
  children: React.ReactNode[];
  autoPlay?: boolean;
  interval?: number;
  showIndicators?: boolean;
  onPageChange?: (index: number) => void;
}

export const Carousel: React.FC<CarouselProps> = ({
  children,
  autoPlay = false,
  interval = 3000,
  showIndicators = true,
  onPageChange,
}) => {
  const { theme } = useTheme();
  const scrollViewRef = useRef<ScrollView>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const screenWidth = Dimensions.get('window').width;

  React.useEffect(() => {
    if (autoPlay && children.length > 1) {
      const timer = setInterval(() => {
        const nextIndex = (currentIndex + 1) % children.length;
        scrollViewRef.current?.scrollTo({
          x: nextIndex * screenWidth,
          animated: true,
        });
        setCurrentIndex(nextIndex);
        onPageChange?.(nextIndex);
      }, interval);

      return () => clearInterval(timer);
    }
  }, [autoPlay, interval, currentIndex, children.length, screenWidth, onPageChange]);

  const handleScroll = (event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const offsetX = event.nativeEvent.contentOffset.x;
    const index = Math.round(offsetX / screenWidth);
    if (index !== currentIndex) {
      setCurrentIndex(index);
      onPageChange?.(index);
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView
        ref={scrollViewRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        style={styles.scrollView}
      >
        {children.map((child, index) => (
          <View key={index} style={[styles.slide, { width: screenWidth }]}>
            {child}
          </View>
        ))}
      </ScrollView>
      {showIndicators && children.length > 1 && (
        <View style={styles.indicators}>
          {children.map((_, index) => (
            <View
              key={index}
              style={[
                styles.indicator,
                {
                  backgroundColor: index === currentIndex ? theme.primary : theme.border,
                },
              ]}
            />
          ))}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  scrollView: {
    flexGrow: 0,
  },
  slide: {
    flex: 1,
  },
  indicators: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.sm,
  },
  indicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
});

