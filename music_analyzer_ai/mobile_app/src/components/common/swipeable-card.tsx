import React, { useRef } from 'react';
import { View, StyleSheet, Animated, PanResponder } from 'react-native';
import { Card } from './card';
import { COLORS } from '../../constants/config';

interface SwipeableCardProps {
  children: React.ReactNode;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  swipeThreshold?: number;
  disabled?: boolean;
}

/**
 * Swipeable card component
 * Supports left and right swipe gestures
 */
export function SwipeableCard({
  children,
  onSwipeLeft,
  onSwipeRight,
  swipeThreshold = 100,
  disabled = false,
}: SwipeableCardProps) {
  const translateX = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(1)).current;

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => !disabled,
      onMoveShouldSetPanResponder: () => !disabled,
      onPanResponderMove: (_, gestureState) => {
        if (!disabled) {
          translateX.setValue(gestureState.dx);
          opacity.setValue(1 - Math.abs(gestureState.dx) / swipeThreshold);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        if (disabled) return;

        const { dx } = gestureState;

        if (Math.abs(dx) > swipeThreshold) {
          if (dx > 0 && onSwipeRight) {
            Animated.parallel([
              Animated.timing(translateX, {
                toValue: 1000,
                duration: 200,
                useNativeDriver: true,
              }),
              Animated.timing(opacity, {
                toValue: 0,
                duration: 200,
                useNativeDriver: true,
              }),
            ]).start(() => {
              onSwipeRight();
              translateX.setValue(0);
              opacity.setValue(1);
            });
          } else if (dx < 0 && onSwipeLeft) {
            Animated.parallel([
              Animated.timing(translateX, {
                toValue: -1000,
                duration: 200,
                useNativeDriver: true,
              }),
              Animated.timing(opacity, {
                toValue: 0,
                duration: 200,
                useNativeDriver: true,
              }),
            ]).start(() => {
              onSwipeLeft();
              translateX.setValue(0);
              opacity.setValue(1);
            });
          }
        } else {
          Animated.parallel([
            Animated.spring(translateX, {
              toValue: 0,
              useNativeDriver: true,
            }),
            Animated.spring(opacity, {
              toValue: 1,
              useNativeDriver: true,
            }),
          ]).start();
        }
      },
    })
  ).current;

  return (
    <Animated.View
      style={[
        styles.container,
        {
          transform: [{ translateX }],
          opacity,
        },
      ]}
      {...panResponder.panHandlers}
    >
      <Card>{children}</Card>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
  },
});

