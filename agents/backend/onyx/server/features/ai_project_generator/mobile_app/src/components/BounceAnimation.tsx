import React, { useEffect, useRef } from 'react';
import { Animated, ViewStyle } from 'react-native';

interface BounceAnimationProps {
  children: React.ReactNode;
  duration?: number;
  delay?: number;
  style?: ViewStyle;
}

export const BounceAnimation: React.FC<BounceAnimationProps> = ({
  children,
  duration = 600,
  delay = 0,
  style,
}) => {
  const bounceAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.delay(delay),
      Animated.spring(bounceAnim, {
        toValue: 1,
        tension: 10,
        friction: 3,
        useNativeDriver: true,
      }),
    ]).start();
  }, [bounceAnim, delay]);

  const translateY = bounceAnim.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [-50, 10, 0],
  });

  return (
    <Animated.View
      style={[
        {
          transform: [{ translateY }, { scale: bounceAnim }],
        },
        style,
      ]}
    >
      {children}
    </Animated.View>
  );
};

