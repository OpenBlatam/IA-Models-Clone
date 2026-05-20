import React, { memo, ReactNode } from 'react';
import { View, ViewProps } from 'react-native';
import * as Animatable from 'react-native-animatable';

interface AnimatableViewProps extends ViewProps {
  children: ReactNode;
  animation?: Animatable.Animation;
  duration?: number;
  delay?: number;
  iterationCount?: number | 'infinite';
}

function AnimatableViewComponent({
  children,
  animation = 'fadeIn',
  duration = 300,
  delay = 0,
  iterationCount = 1,
  ...props
}: AnimatableViewProps): JSX.Element {
  return (
    <Animatable.View
      animation={animation}
      duration={duration}
      delay={delay}
      iterationCount={iterationCount}
      {...props}
    >
      {children}
    </Animatable.View>
  );
}

export const AnimatableView = memo(AnimatableViewComponent);

