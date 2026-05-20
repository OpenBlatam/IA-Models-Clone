import { ReactNode, useEffect } from 'react';
import { ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withDelay,
  Easing,
} from 'react-native-reanimated';

interface FadeInViewProps {
  children: ReactNode;
  delay?: number;
  duration?: number;
  style?: ViewStyle;
  from?: 'top' | 'bottom' | 'left' | 'right' | 'fade';
}

export function FadeInView({
  children,
  delay = 0,
  duration = 300,
  style,
  from = 'fade',
}: FadeInViewProps) {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(from === 'top' ? -20 : from === 'bottom' ? 20 : 0);
  const translateX = useSharedValue(from === 'left' ? -20 : from === 'right' ? 20 : 0);

  useEffect(() => {
    opacity.value = withDelay(
      delay,
      withTiming(1, {
        duration,
        easing: Easing.out(Easing.ease),
      })
    );

    if (from === 'top' || from === 'bottom') {
      translateY.value = withDelay(
        delay,
        withTiming(0, {
          duration,
          easing: Easing.out(Easing.ease),
        })
      );
    }

    if (from === 'left' || from === 'right') {
      translateX.value = withDelay(
        delay,
        withTiming(0, {
          duration,
          easing: Easing.out(Easing.ease),
        })
      );
    }
  }, [delay, duration, from]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
      transform: [
        { translateY: translateY.value },
        { translateX: translateX.value },
      ],
    };
  });

  return (
    <Animated.View style={[style, animatedStyle]}>
      {children}
    </Animated.View>
  );
}


