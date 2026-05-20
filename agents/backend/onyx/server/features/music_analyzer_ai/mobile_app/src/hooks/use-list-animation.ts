import { useRef, useCallback } from 'react';
import { Animated, LayoutAnimation, Platform, UIManager } from 'react-native';

if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

interface UseListAnimationOptions {
  animationType?: 'fade' | 'slide' | 'scale' | 'easeInEaseOut';
  duration?: number;
}

export function useListAnimation(options: UseListAnimationOptions = {}) {
  const { animationType = 'easeInEaseOut', duration = 300 } = options;
  const animatedValues = useRef<Map<string, Animated.Value>>(new Map());

  const getAnimationConfig = useCallback(() => {
    switch (animationType) {
      case 'fade':
        return {
          duration,
          create: {
            type: LayoutAnimation.Types.easeInEaseOut,
            property: LayoutAnimation.Properties.opacity,
          },
      update: {
        type: LayoutAnimation.Types.easeInEaseOut,
      },
    };
      case 'slide':
        return {
          duration,
          create: {
            type: LayoutAnimation.Types.easeInEaseOut,
            property: LayoutAnimation.Properties.opacity,
          },
          update: {
            type: LayoutAnimation.Types.easeInEaseOut,
          },
        };
      case 'scale':
        return {
          duration,
          create: {
            type: LayoutAnimation.Types.spring,
            property: LayoutAnimation.Properties.scaleXY,
            springDamping: 0.7,
          },
          update: {
            type: LayoutAnimation.Types.easeInEaseOut,
          },
        };
      default:
        return {
          duration,
          create: {
            type: LayoutAnimation.Types.easeInEaseOut,
            property: LayoutAnimation.Properties.opacity,
          },
          update: {
            type: LayoutAnimation.Types.easeInEaseOut,
          },
        };
    }
  }, [animationType, duration]);

  const animateListChange = useCallback(() => {
    LayoutAnimation.configureNext(getAnimationConfig());
  }, [getAnimationConfig]);

  const createItemAnimation = useCallback((key: string, delay = 0) => {
    if (!animatedValues.current.has(key)) {
      const animatedValue = new Animated.Value(0);
      animatedValues.current.set(key, animatedValue);

      Animated.timing(animatedValue, {
        toValue: 1,
        duration: duration - delay,
        delay,
        useNativeDriver: true,
      }).start();
    }

    return animatedValues.current.get(key) || new Animated.Value(1);
  }, [duration]);

  const removeItemAnimation = useCallback((key: string) => {
    const animatedValue = animatedValues.current.get(key);
    if (animatedValue) {
      Animated.timing(animatedValue, {
        toValue: 0,
        duration: duration / 2,
        useNativeDriver: true,
      }).start(() => {
        animatedValues.current.delete(key);
      });
    }
  }, [duration]);

  return {
    animateListChange,
    createItemAnimation,
    removeItemAnimation,
    animatedValues: animatedValues.current,
  };
}

