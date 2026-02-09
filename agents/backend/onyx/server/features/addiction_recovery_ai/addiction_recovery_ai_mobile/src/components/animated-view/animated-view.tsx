import React, { ReactNode } from 'react';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  FadeIn,
  FadeOut,
  SlideInDown,
  SlideOutDown,
} from 'react-native-reanimated';
import { ViewStyle } from 'react-native';

interface AnimatedViewProps {
  children: ReactNode;
  style?: ViewStyle;
  delay?: number;
  duration?: number;
  animation?: 'fade' | 'slide' | 'scale';
  onAnimationComplete?: () => void;
}

const AnimatedViewComponent = Animated.View;

export function AnimatedView({
  children,
  style,
  delay = 0,
  duration = 300,
  animation = 'fade',
  onAnimationComplete,
}: AnimatedViewProps): JSX.Element {
  const scale = useSharedValue(animation === 'scale' ? 0 : 1);
  const opacity = useSharedValue(animation === 'fade' ? 0 : 1);

  React.useEffect(() => {
    const timer = setTimeout(() => {
      if (animation === 'scale') {
        scale.value = withSpring(1, { damping: 15 });
      } else if (animation === 'fade') {
        opacity.value = withTiming(1, { duration }, () => {
          onAnimationComplete?.();
        });
      }
    }, delay);

    return () => clearTimeout(timer);
  }, [animation, delay, duration, onAnimationComplete, opacity, scale]);

  const animatedStyle = useAnimatedStyle(() => {
    if (animation === 'scale') {
      return {
        transform: [{ scale: scale.value }],
        opacity: opacity.value,
      };
    }
    if (animation === 'fade') {
      return {
        opacity: opacity.value,
      };
    }
    return {};
  });

  if (animation === 'fade') {
    return (
      <AnimatedViewComponent
        entering={FadeIn.delay(delay).duration(duration)}
        exiting={FadeOut.duration(duration)}
        style={[animatedStyle, style]}
      >
        {children}
      </AnimatedViewComponent>
    );
  }

  if (animation === 'slide') {
    return (
      <AnimatedViewComponent
        entering={SlideInDown.delay(delay).duration(duration)}
        exiting={SlideOutDown.duration(duration)}
        style={style}
      >
        {children}
      </AnimatedViewComponent>
    );
  }

  return (
    <AnimatedViewComponent style={[animatedStyle, style]}>
      {children}
    </AnimatedViewComponent>
  );
}

