import React, { useRef, useState } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Dimensions,
  NativeScrollEvent,
  NativeSyntheticEvent,
} from 'react-native';
import { useTheme } from '../context/ThemeContext';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface CarouselProps {
  children: React.ReactNode[];
  itemWidth?: number;
  spacing?: number;
  onPageChange?: (index: number) => void;
  showIndicators?: boolean;
  autoPlay?: boolean;
  autoPlayInterval?: number;
}

const Carousel: React.FC<CarouselProps> = ({
  children,
  itemWidth = SCREEN_WIDTH * 0.9,
  spacing = 10,
  onPageChange,
  showIndicators = true,
  autoPlay = false,
  autoPlayInterval = 3000,
}) => {
  const { colors } = useTheme();
  const scrollViewRef = useRef<ScrollView>(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleScroll = (event: NativeSyntheticEvent<NativeScrollEvent>) => {
    const offsetX = event.nativeEvent.contentOffset.x;
    const index = Math.round(offsetX / (itemWidth + spacing));
    
    if (index !== currentIndex) {
      setCurrentIndex(index);
      onPageChange?.(index);
    }
  };

  React.useEffect(() => {
    if (autoPlay && children.length > 1) {
      const interval = setInterval(() => {
        const nextIndex = (currentIndex + 1) % children.length;
        scrollViewRef.current?.scrollTo({
          x: nextIndex * (itemWidth + spacing),
          animated: true,
        });
        setCurrentIndex(nextIndex);
        onPageChange?.(nextIndex);
      }, autoPlayInterval);

      return () => clearInterval(interval);
    }
  }, [autoPlay, autoPlayInterval, currentIndex, children.length, itemWidth, spacing, onPageChange]);

  return (
    <View style={styles.container}>
      <ScrollView
        ref={scrollViewRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        snapToInterval={itemWidth + spacing}
        decelerationRate="fast"
        contentContainerStyle={[
          styles.contentContainer,
          { paddingHorizontal: (SCREEN_WIDTH - itemWidth) / 2 },
        ]}
      >
        {children.map((child, index) => (
          <View
            key={index}
            style={[
              styles.item,
              {
                width: itemWidth,
                marginRight: index < children.length - 1 ? spacing : 0,
              },
            ]}
          >
            {child}
          </View>
        ))}
      </ScrollView>

      {showIndicators && (
        <View style={styles.indicators}>
          {children.map((_, index) => (
            <View
              key={index}
              style={[
                styles.indicator,
                {
                  backgroundColor:
                    index === currentIndex ? colors.primary : colors.border,
                  width: index === currentIndex ? 24 : 8,
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
    width: '100%',
  },
  contentContainer: {
    alignItems: 'center',
  },
  item: {
    flex: 1,
  },
  indicators: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 16,
    gap: 8,
  },
  indicator: {
    height: 8,
    borderRadius: 4,
    transition: 'all 0.3s',
  },
});

export default Carousel;

