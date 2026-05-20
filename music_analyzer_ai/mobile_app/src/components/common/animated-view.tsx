import React, { ReactNode } from 'react';
import Animated, {
  FadeIn,
  FadeOut,
  SlideInDown,
  SlideOutDown,
} from 'react-native-reanimated';

interface AnimatedViewProps {
  children: ReactNode;
  delay?: number;
  duration?: number;
  style?: unknown;
}

export function AnimatedView({
  children,
  delay = 0,
  duration = 300,
  style,
}: AnimatedViewProps) {
  return (
    <Animated.View
      entering={FadeIn.delay(delay).duration(duration)}
      exiting={FadeOut.duration(duration)}
      style={style}
    >
      {children}
    </Animated.View>
  );
}

interface SlideUpViewProps {
  children: ReactNode;
  delay?: number;
  duration?: number;
  style?: unknown;
}

export function SlideUpView({
  children,
  delay = 0,
  duration = 400,
  style,
}: SlideUpViewProps) {
  return (
    <Animated.View
      entering={SlideInDown.delay(delay).duration(duration)}
      exiting={SlideOutDown.duration(duration)}
      style={style}
    >
      {children}
    </Animated.View>
  );
}

