import React, { useState } from 'react';
import { View, TouchableOpacity, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import { useTheme } from '../context/ThemeContext';

interface RippleEffectProps {
  children: React.ReactNode;
  onPress?: () => void;
  rippleColor?: string;
  style?: ViewStyle;
}

const RippleEffect: React.FC<RippleEffectProps> = ({
  children,
  onPress,
  rippleColor,
  style,
}) => {
  const { colors } = useTheme();
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number }>>([]);
  const rippleColorValue = rippleColor || colors.primary;

  const handlePress = (event: any) => {
    const { locationX, locationY } = event.nativeEvent;
    const newRipple = {
      id: Date.now(),
      x: locationX,
      y: locationY,
    };
    setRipples((prev) => [...prev, newRipple]);
    onPress?.();
    
    setTimeout(() => {
      setRipples((prev) => prev.filter((r) => r.id !== newRipple.id));
    }, 600);
  };

  return (
    <TouchableOpacity
      onPress={handlePress}
      activeOpacity={1}
      style={[styles.container, style]}
    >
      {children}
      {ripples.map((ripple) => (
        <Ripple
          key={ripple.id}
          x={ripple.x}
          y={ripple.y}
          color={rippleColorValue}
        />
      ))}
    </TouchableOpacity>
  );
};

const Ripple: React.FC<{ x: number; y: number; color: string }> = ({
  x,
  y,
  color,
}) => {
  const scale = useSharedValue(0);
  const opacity = useSharedValue(0.5);

  React.useEffect(() => {
    scale.value = withSpring(4, { damping: 15, stiffness: 100 });
    opacity.value = withTiming(0, { duration: 600 });
  }, [scale, opacity]);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: scale.value }],
      opacity: opacity.value,
    };
  });

  return (
    <Animated.View
      style={[
        styles.ripple,
        {
          left: x - 20,
          top: y - 20,
          backgroundColor: color,
        },
        animatedStyle,
      ]}
    />
  );
};

const styles = StyleSheet.create({
  container: {
    overflow: 'hidden',
    position: 'relative',
  },
  ripple: {
    position: 'absolute',
    width: 40,
    height: 40,
    borderRadius: 20,
  },
});

export default RippleEffect;

