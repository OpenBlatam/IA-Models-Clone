import React, { useEffect } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withRepeat,
  withSequence,
  Easing,
} from 'react-native-reanimated';

interface AnimatedViewProps {
  children: React.ReactNode;
  style?: ViewStyle;
  animation?: 'fadeIn' | 'slideIn' | 'scaleIn' | 'bounce' | 'pulse' | 'shake';
  duration?: number;
  delay?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
}

const AnimatedView: React.FC<AnimatedViewProps> = ({
  children,
  style,
  animation = 'fadeIn',
  duration = 300,
  delay = 0,
  direction = 'up',
}) => {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(0);
  const translateX = useSharedValue(0);
  const scale = useSharedValue(1);

  useEffect(() => {
    const getInitialValues = () => {
      switch (animation) {
        case 'fadeIn':
          return { opacity: 0, translateY: 0, translateX: 0, scale: 1 };
        case 'slideIn':
          const slideDistance = 50;
          return {
            opacity: 0,
            translateY: direction === 'up' ? slideDistance : direction === 'down' ? -slideDistance : 0,
            translateX: direction === 'left' ? slideDistance : direction === 'right' ? -slideDistance : 0,
            scale: 1,
          };
        case 'scaleIn':
          return { opacity: 0, translateY: 0, translateX: 0, scale: 0.8 };
        case 'bounce':
          return { opacity: 0, translateY: -30, translateX: 0, scale: 0.9 };
        case 'pulse':
          return { opacity: 1, translateY: 0, translateX: 0, scale: 1 };
        case 'shake':
          return { opacity: 1, translateY: 0, translateX: 0, scale: 1 };
        default:
          return { opacity: 0, translateY: 0, translateX: 0, scale: 1 };
      }
    };

    const initial = getInitialValues();
    opacity.value = initial.opacity;
    translateY.value = initial.translateY;
    translateX.value = initial.translateX;
    scale.value = initial.scale;

    const animate = () => {
      const config = {
        duration,
        easing: Easing.out(Easing.ease),
      };

      switch (animation) {
        case 'fadeIn':
          opacity.value = withTiming(1, config);
          break;
        case 'slideIn':
          opacity.value = withTiming(1, config);
          translateY.value = withSpring(0, { damping: 15, stiffness: 100 });
          translateX.value = withSpring(0, { damping: 15, stiffness: 100 });
          break;
        case 'scaleIn':
          opacity.value = withTiming(1, config);
          scale.value = withSpring(1, { damping: 15, stiffness: 100 });
          break;
        case 'bounce':
          opacity.value = withTiming(1, config);
          translateY.value = withSpring(0, { damping: 8, stiffness: 100 });
          scale.value = withSpring(1, { damping: 8, stiffness: 100 });
          break;
        case 'pulse':
          scale.value = withRepeat(
            withSequence(
              withTiming(1.05, { duration: 500 }),
              withTiming(1, { duration: 500 })
            ),
            -1,
            true
          );
          break;
        case 'shake':
          translateX.value = withRepeat(
            withSequence(
              withTiming(-10, { duration: 50 }),
              withTiming(10, { duration: 50 }),
              withTiming(-10, { duration: 50 }),
              withTiming(10, { duration: 50 }),
              withTiming(0, { duration: 50 })
            ),
            1
          );
          break;
      }
    };

    const timer = setTimeout(animate, delay);
    return () => clearTimeout(timer);
  }, [animation, duration, delay, direction]);

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

export default AnimatedView;

