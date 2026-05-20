import React, { useState, useRef } from 'react';
import {
  View,
  Image,
  StyleSheet,
  ScrollView,
  Dimensions,
  TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface ImageCarouselProps {
  images: string[];
  height?: number;
  showIndicators?: boolean;
  autoPlay?: boolean;
  autoPlayInterval?: number;
}

const ImageCarousel: React.FC<ImageCarouselProps> = ({
  images,
  height = 200,
  showIndicators = true,
  autoPlay = false,
  autoPlayInterval = 3000,
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollViewRef = useRef<ScrollView>(null);
  const screenWidth = Dimensions.get('window').width;

  React.useEffect(() => {
    if (autoPlay && images.length > 1) {
      const interval = setInterval(() => {
        const nextIndex = (currentIndex + 1) % images.length;
        scrollViewRef.current?.scrollTo({
          x: nextIndex * screenWidth,
          animated: true,
        });
        setCurrentIndex(nextIndex);
      }, autoPlayInterval);

      return () => clearInterval(interval);
    }
  }, [currentIndex, autoPlay, images.length, autoPlayInterval, screenWidth]);

  const handleScroll = (event: any) => {
    const contentOffsetX = event.nativeEvent.contentOffset.x;
    const index = Math.round(contentOffsetX / screenWidth);
    setCurrentIndex(index);
  };

  const goToIndex = (index: number) => {
    scrollViewRef.current?.scrollTo({
      x: index * screenWidth,
      animated: true,
    });
    setCurrentIndex(index);
  };

  if (images.length === 0) return null;

  return (
    <View style={styles.container}>
      <ScrollView
        ref={scrollViewRef}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        style={[styles.scrollView, { height }]}
      >
        {images.map((uri, index) => (
          <Image
            key={index}
            source={{ uri }}
            style={[styles.image, { width: screenWidth, height }]}
            resizeMode="cover"
          />
        ))}
      </ScrollView>

      {showIndicators && images.length > 1 && (
        <View style={styles.indicators}>
          {images.map((_, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.indicator,
                index === currentIndex && styles.indicatorActive,
              ]}
              onPress={() => goToIndex(index)}
            />
          ))}
        </View>
      )}

      {images.length > 1 && (
        <>
          {currentIndex > 0 && (
            <TouchableOpacity
              style={[styles.navButton, styles.prevButton]}
              onPress={() => goToIndex(currentIndex - 1)}
            >
              <Ionicons name="chevron-back" size={24} color="#fff" />
            </TouchableOpacity>
          )}
          {currentIndex < images.length - 1 && (
            <TouchableOpacity
              style={[styles.navButton, styles.nextButton]}
              onPress={() => goToIndex(currentIndex + 1)}
            >
              <Ionicons name="chevron-forward" size={24} color="#fff" />
            </TouchableOpacity>
          )}
        </>
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
  image: {
    resizeMode: 'cover',
  },
  indicators: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'absolute',
    bottom: 16,
    left: 0,
    right: 0,
  },
  indicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    marginHorizontal: 4,
  },
  indicatorActive: {
    backgroundColor: '#fff',
    width: 24,
  },
  navButton: {
    position: 'absolute',
    top: '50%',
    transform: [{ translateY: -20 }],
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 20,
    padding: 8,
  },
  prevButton: {
    left: 16,
  },
  nextButton: {
    right: 16,
  },
});

export default ImageCarousel;

