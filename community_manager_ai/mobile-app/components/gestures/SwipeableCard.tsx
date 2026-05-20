import { ReactNode } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';

interface SwipeableCardProps {
  children: ReactNode;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  swipeThreshold?: number;
  style?: ViewStyle;
}

export function SwipeableCard({
  children,
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
  swipeThreshold = 100,
  style,
}: SwipeableCardProps) {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const opacity = useSharedValue(1);

  const panGesture = Gesture.Pan()
    .onUpdate((event) => {
      translateX.value = event.translationX;
      translateY.value = event.translationY;
      
      // Fade out as user swipes
      const distance = Math.sqrt(
        event.translationX ** 2 + event.translationY ** 2
      );
      opacity.value = Math.max(0, 1 - distance / swipeThreshold);
    })
    .onEnd((event) => {
      const absX = Math.abs(event.translationX);
      const absY = Math.abs(event.translationY);

      if (absX > swipeThreshold || absY > swipeThreshold) {
        if (absX > absY) {
          // Horizontal swipe
          if (event.translationX > 0 && onSwipeRight) {
            runOnJS(onSwipeRight)();
          } else if (event.translationX < 0 && onSwipeLeft) {
            runOnJS(onSwipeLeft)();
          }
        } else {
          // Vertical swipe
          if (event.translationY > 0 && onSwipeDown) {
            runOnJS(onSwipeDown)();
          } else if (event.translationY < 0 && onSwipeUp) {
            runOnJS(onSwipeUp)();
          }
        }
      }

      // Reset position
      translateX.value = withSpring(0);
      translateY.value = withSpring(0);
      opacity.value = withSpring(1);
    });

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
      ],
      opacity: opacity.value,
    };
  });

  return (
    <GestureDetector gesture={panGesture}>
      <Animated.View style={[styles.card, style, animatedStyle]}>
        {children}
      </Animated.View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  card: {
    width: '100%',
  },
});


