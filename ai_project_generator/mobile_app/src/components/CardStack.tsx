import React, { useState } from 'react';
import { View, StyleSheet, PanResponder, Animated, Dimensions } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface CardStackProps {
  cards: React.ReactNode[];
  onSwipeLeft?: (index: number) => void;
  onSwipeRight?: (index: number) => void;
  onSwipeComplete?: (index: number) => void;
  threshold?: number;
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export const CardStack: React.FC<CardStackProps> = ({
  cards,
  onSwipeLeft,
  onSwipeRight,
  onSwipeComplete,
  threshold = 100,
}) => {
  const { theme } = useTheme();
  const [currentIndex, setCurrentIndex] = useState(0);
  const position = React.useRef(new Animated.ValueXY()).current;

  const panResponder = React.useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: (_, gestureState) => {
        position.setValue({ x: gestureState.dx, y: gestureState.dy });
      },
      onPanResponderRelease: (_, gestureState) => {
        if (Math.abs(gestureState.dx) > threshold) {
          if (gestureState.dx > 0) {
            handleSwipeRight();
          } else {
            handleSwipeLeft();
          }
        } else {
          Animated.spring(position, {
            toValue: { x: 0, y: 0 },
            useNativeDriver: true,
          }).start();
        }
      },
    })
  ).current;

  const handleSwipeLeft = () => {
    hapticFeedback.selection();
    Animated.timing(position, {
      toValue: { x: -SCREEN_WIDTH, y: 0 },
      duration: 300,
      useNativeDriver: true,
    }).start(() => {
      onSwipeLeft?.(currentIndex);
      nextCard();
    });
  };

  const handleSwipeRight = () => {
    hapticFeedback.selection();
    Animated.timing(position, {
      toValue: { x: SCREEN_WIDTH, y: 0 },
      duration: 300,
      useNativeDriver: true,
    }).start(() => {
      onSwipeRight?.(currentIndex);
      nextCard();
    });
  };

  const nextCard = () => {
    if (currentIndex < cards.length - 1) {
      setCurrentIndex(currentIndex + 1);
      position.setValue({ x: 0, y: 0 });
      onSwipeComplete?.(currentIndex);
    }
  };

  const rotate = position.x.interpolate({
    inputRange: [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
    outputRange: ['-30deg', '0deg', '30deg'],
  });

  const opacity = position.x.interpolate({
    inputRange: [-SCREEN_WIDTH, 0, SCREEN_WIDTH],
    outputRange: [0, 1, 0],
  });

  if (currentIndex >= cards.length) {
    return null;
  }

  return (
    <View style={styles.container}>
      {cards.slice(currentIndex, currentIndex + 2).map((card, index) => {
        const isTop = index === 0;
        const scale = isTop ? 1 : 0.95;
        const zIndex = isTop ? 2 : 1;

        return (
          <Animated.View
            key={currentIndex + index}
            style={[
              styles.card,
              {
                backgroundColor: theme.surface,
                borderRadius: borderRadius.lg,
                transform: isTop
                  ? [
                      { translateX: position.x },
                      { translateY: position.y },
                      { rotate },
                    ]
                  : [{ scale }],
                opacity: isTop ? opacity : 1,
                zIndex,
              },
            ]}
            {...(isTop ? panResponder.panHandlers : {})}
          >
            {card}
          </Animated.View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    position: 'absolute',
    width: SCREEN_WIDTH - spacing.xl * 2,
    height: '70%',
    padding: spacing.lg,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
});

