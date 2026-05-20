import React, { useRef } from 'react';
import { View, StyleSheet, Dimensions, PanResponder } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;

interface CardStackProps {
  children: React.ReactNode[];
  onSwipeLeft?: (index: number) => void;
  onSwipeRight?: (index: number) => void;
  onSwipeUp?: (index: number) => void;
  onSwipeDown?: (index: number) => void;
}

const CardStack: React.FC<CardStackProps> = ({
  children,
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
}) => {
  const { colors } = useTheme();
  const [currentIndex, setCurrentIndex] = React.useState(0);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const rotate = useSharedValue(0);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        // Reset values
      },
      onPanResponderMove: (_, gestureState) => {
        translateX.value = gestureState.dx;
        translateY.value = gestureState.dy;
        rotate.value = (gestureState.dx / SCREEN_WIDTH) * 10;
      },
      onPanResponderRelease: (_, gestureState) => {
        const { dx, dy, vx, vy } = gestureState;
        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);

        if (absDx > absDy) {
          // Horizontal swipe
          if (absDx > SWIPE_THRESHOLD || Math.abs(vx) > 0.5) {
            if (dx > 0) {
              runOnJS(handleSwipeRight)();
            } else {
              runOnJS(handleSwipeLeft)();
            }
          } else {
            resetPosition();
          }
        } else {
          // Vertical swipe
          if (absDy > SWIPE_THRESHOLD || Math.abs(vy) > 0.5) {
            if (dy > 0) {
              runOnJS(handleSwipeDown)();
            } else {
              runOnJS(handleSwipeUp)();
            }
          } else {
            resetPosition();
          }
        }
      },
    })
  ).current;

  const handleSwipeLeft = () => {
    onSwipeLeft?.(currentIndex);
    nextCard();
  };

  const handleSwipeRight = () => {
    onSwipeRight?.(currentIndex);
    nextCard();
  };

  const handleSwipeUp = () => {
    onSwipeUp?.(currentIndex);
    nextCard();
  };

  const handleSwipeDown = () => {
    onSwipeDown?.(currentIndex);
    nextCard();
  };

  const nextCard = () => {
    if (currentIndex < children.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
    resetPosition();
  };

  const resetPosition = () => {
    translateX.value = withSpring(0);
    translateY.value = withSpring(0);
    rotate.value = withSpring(0);
  };

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { rotate: `${rotate.value}deg` },
      ],
    };
  });

  if (currentIndex >= children.length) {
    return null;
  }

  return (
    <View style={styles.container}>
      {children.slice(currentIndex, currentIndex + 3).map((child, index) => {
        const isTop = index === 0;
        const zIndex = 3 - index;
        const scale = 1 - index * 0.05;
        const offsetY = index * 8;

        return (
          <Animated.View
            key={currentIndex + index}
            style={[
              styles.card,
              {
                backgroundColor: colors.card,
                zIndex,
                transform: [
                  { scale },
                  { translateY: offsetY },
                ],
              },
              isTop && animatedStyle,
            ]}
            {...(isTop ? panResponder.panHandlers : {})}
          >
            {child}
          </Animated.View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: 400,
    position: 'relative',
  },
  card: {
    position: 'absolute',
    width: '90%',
    height: '100%',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 8,
    alignSelf: 'center',
  },
});

export default CardStack;

