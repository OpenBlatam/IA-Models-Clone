import { ReactNode, useEffect } from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  FadeIn,
  FadeOut,
  SlideInDown,
  SlideOutDown,
} from 'react-native-reanimated';

interface AnimatedViewProps {
  children: ReactNode;
  style?: ViewStyle;
  animation?: 'fade' | 'slide' | 'scale' | 'none';
  delay?: number;
  duration?: number;
}

export function AnimatedView({
  children,
  style,
  animation = 'fade',
  delay = 0,
  duration = 300,
}: AnimatedViewProps) {
  const opacity = useSharedValue(0);
  const scale = useSharedValue(0.9);

  useEffect(() => {
    if (animation === 'fade') {
      opacity.value = withTiming(1, { duration });
    } else if (animation === 'scale') {
      opacity.value = withTiming(1, { duration });
      scale.value = withSpring(1, { damping: 15 });
    }
  }, [animation, duration, opacity, scale]);

  const animatedStyle = useAnimatedStyle(() => {
    if (animation === 'fade') {
      return { opacity: opacity.value };
    }
    if (animation === 'scale') {
      return { opacity: opacity.value, transform: [{ scale: scale.value }] };
    }
    return {};
  });

  if (animation === 'none') {
    return <View style={style}>{children}</View>;
  }

  return (
    <Animated.View
      entering={animation === 'fade' ? FadeIn.delay(delay) : SlideInDown.delay(delay)}
      exiting={animation === 'fade' ? FadeOut : SlideOutDown}
      style={[style, animatedStyle]}
    >
      {children}
    </Animated.View>
  );
}

