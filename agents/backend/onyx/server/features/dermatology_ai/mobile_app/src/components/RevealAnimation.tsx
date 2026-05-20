import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withDelay,
} from 'react-native-reanimated';

interface RevealAnimationProps {
  children: React.ReactNode;
  direction?: 'up' | 'down' | 'left' | 'right' | 'fade' | 'scale';
  delay?: number;
  duration?: number;
  style?: ViewStyle;
}

const RevealAnimation: React.FC<RevealAnimationProps> = ({
  children,
  direction = 'fade',
  delay = 0,
  duration = 500,
  style,
}) => {
  const opacity = useSharedValue(0);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const scale = useSharedValue(0.8);

  useEffect(() => {
    const getInitialValues = () => {
      switch (direction) {
        case 'up':
          return { translateY: 50, translateX: 0, scale: 1 };
        case 'down':
          return { translateY: -50, translateX: 0, scale: 1 };
        case 'left':
          return { translateY: 0, translateX: 50, scale: 1 };
        case 'right':
          return { translateY: 0, translateX: -50, scale: 1 };
        case 'scale':
          return { translateY: 0, translateX: 0, scale: 0.8 };
        default:
          return { translateY: 0, translateX: 0, scale: 1 };
      }
    };

    const initial = getInitialValues();
    translateY.value = initial.translateY;
    translateX.value = initial.translateX;
    scale.value = initial.scale;

    opacity.value = withDelay(
      delay,
      withTiming(1, { duration })
    );
    translateY.value = withDelay(
      delay,
      withSpring(0, { damping: 15, stiffness: 100 })
    );
    translateX.value = withDelay(
      delay,
      withSpring(0, { damping: 15, stiffness: 100 })
    );
    scale.value = withDelay(
      delay,
      withSpring(1, { damping: 15, stiffness: 100 })
    );
  }, [direction, delay, duration, opacity, translateY, translateX, scale]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [
        { translateY: translateY.value },
        { translateX: translateX.value },
        { scale: scale.value },
      ],
    };
  });

  return (
    <Animated.View style={[style, animatedStyle]}>
      {children}
    </Animated.View>
  );
};

export default RevealAnimation;

