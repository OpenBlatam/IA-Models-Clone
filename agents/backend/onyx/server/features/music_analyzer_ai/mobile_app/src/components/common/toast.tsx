import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  runOnJS,
} from 'react-native-reanimated';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';
import { useHapticFeedback } from '../../hooks/use-haptic-feedback';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'info' | 'warning';
  visible: boolean;
  duration?: number;
  onHide: () => void;
}

const AnimatedView = Animated.createAnimatedComponent(View);

export function Toast({
  message,
  type = 'info',
  visible,
  duration = 3000,
  onHide,
}: ToastProps) {
  const haptics = useHapticFeedback();
  const translateY = useSharedValue(-100);
  const opacity = useSharedValue(0);

  useEffect(() => {
    if (visible) {
      haptics.light();
      translateY.value = withSpring(0);
      opacity.value = withTiming(1);

      const timer = setTimeout(() => {
        translateY.value = withSpring(-100);
        opacity.value = withTiming(0, {}, () => {
          runOnJS(onHide)();
        });
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [visible, duration, translateY, opacity, onHide, haptics]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
    opacity: opacity.value,
  }));

  if (!visible) return null;

  const typeColors = {
    success: COLORS.success,
    error: COLORS.error,
    info: COLORS.info,
    warning: COLORS.warning,
  };

  return (
    <AnimatedView style={[styles.container, animatedStyle]}>
      <View style={[styles.toast, { backgroundColor: typeColors[type] }]}>
        <Text style={styles.message}>{message}</Text>
      </View>
    </AnimatedView>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 60,
    left: SPACING.md,
    right: SPACING.md,
    zIndex: 9999,
  },
  toast: {
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  message: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    fontWeight: '600',
    textAlign: 'center',
  },
});

