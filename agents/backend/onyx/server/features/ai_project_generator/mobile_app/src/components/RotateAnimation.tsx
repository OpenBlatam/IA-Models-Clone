import React, { useEffect, useRef } from 'react';
import { Animated, ViewStyle } from 'react-native';

interface RotateAnimationProps {
  children: React.ReactNode;
  duration?: number;
  loop?: boolean;
  style?: ViewStyle;
}

export const RotateAnimation: React.FC<RotateAnimationProps> = ({
  children,
  duration = 2000,
  loop = false,
  style,
}) => {
  const rotateAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.timing(rotateAnim, {
      toValue: 1,
      duration,
      useNativeDriver: true,
    });

    if (loop) {
      Animated.loop(animation).start();
    } else {
      animation.start();
    }
  }, [rotateAnim, duration, loop]);

  const rotate = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <Animated.View
      style={[
        {
          transform: [{ rotate }],
        },
        style,
      ]}
    >
      {children}
    </Animated.View>
  );
};

