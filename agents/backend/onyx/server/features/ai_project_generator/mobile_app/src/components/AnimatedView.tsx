import React, { useEffect, useRef } from 'react';
import { Animated, View, ViewStyle } from 'react-native';

interface AnimatedViewProps {
  children: React.ReactNode;
  style?: ViewStyle;
  animation?: 'fadeIn' | 'slideUp' | 'slideDown' | 'scale' | 'none';
  duration?: number;
  delay?: number;
}

export const AnimatedView: React.FC<AnimatedViewProps> = ({
  children,
  style,
  animation = 'fadeIn',
  duration = 300,
  delay = 0,
}) => {
  const opacity = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(20)).current;
  const scale = useRef(new Animated.Value(0.95)).current;

  useEffect(() => {
    if (animation === 'none') {
      opacity.setValue(1);
      return;
    }

    const animations: Animated.CompositeAnimation[] = [];

    if (animation === 'fadeIn' || animation === 'slideUp' || animation === 'slideDown') {
      animations.push(
        Animated.timing(opacity, {
          toValue: 1,
          duration,
          delay,
          useNativeDriver: true,
        })
      );
    }

    if (animation === 'slideUp') {
      animations.push(
        Animated.timing(translateY, {
          toValue: 0,
          duration,
          delay,
          useNativeDriver: true,
        })
      );
    }

    if (animation === 'slideDown') {
      animations.push(
        Animated.timing(translateY, {
          toValue: 0,
          duration,
          delay,
          useNativeDriver: true,
        })
      );
    }

    if (animation === 'scale') {
      animations.push(
        Animated.timing(scale, {
          toValue: 1,
          duration,
          delay,
          useNativeDriver: true,
        })
      );
      animations.push(
        Animated.timing(opacity, {
          toValue: 1,
          duration,
          delay,
          useNativeDriver: true,
        })
      );
    }

    Animated.parallel(animations).start();
  }, [animation, duration, delay]);

  const getTransform = () => {
    const transforms: any[] = [];
    if (animation === 'slideUp' || animation === 'slideDown') {
      transforms.push({ translateY });
    }
    if (animation === 'scale') {
      transforms.push({ scale });
    }
    return transforms;
  };

  return (
    <Animated.View
      style={[
        style,
        {
          opacity: animation === 'none' ? 1 : opacity,
          transform: getTransform(),
        },
      ]}
    >
      {children}
    </Animated.View>
  );
};

