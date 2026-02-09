import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  runOnJS,
} from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface SnackbarProps {
  message: string;
  visible: boolean;
  duration?: number;
  actionLabel?: string;
  onAction?: () => void;
  onDismiss: () => void;
  type?: 'default' | 'success' | 'error' | 'warning' | 'info';
}

const AnimatedView = Animated.createAnimatedComponent(View);

/**
 * Snackbar component for temporary messages
 * Follows Material Design guidelines
 */
export function Snackbar({
  message,
  visible,
  duration = 4000,
  actionLabel,
  onAction,
  onDismiss,
  type = 'default',
}: SnackbarProps) {
  const translateY = useSharedValue(100);
  const opacity = useSharedValue(0);

  useEffect(() => {
    if (visible) {
      translateY.value = withSpring(0);
      opacity.value = withTiming(1);

      const timer = setTimeout(() => {
        translateY.value = withSpring(100);
        opacity.value = withTiming(0, {}, () => {
          runOnJS(onDismiss)();
        });
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [visible, duration, translateY, opacity, onDismiss]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
    opacity: opacity.value,
  }));

  if (!visible) return null;

  const typeColors = {
    default: COLORS.surface,
    success: COLORS.success,
    error: COLORS.error,
    warning: COLORS.warning,
    info: COLORS.info,
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']} pointerEvents="box-none">
      <AnimatedView
        style={[
          styles.snackbar,
          { backgroundColor: typeColors[type] },
          animatedStyle,
        ]}
      >
        <Text style={styles.message}>{message}</Text>
        {actionLabel && onAction && (
          <TouchableOpacity onPress={onAction} style={styles.action}>
            <Text style={styles.actionText}>{actionLabel}</Text>
          </TouchableOpacity>
        )}
      </AnimatedView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    zIndex: 10000,
    pointerEvents: 'box-none',
  },
  snackbar: {
    margin: SPACING.md,
    padding: SPACING.md,
    borderRadius: BORDER_RADIUS.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 6,
  },
  message: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    flex: 1,
  },
  action: {
    marginLeft: SPACING.md,
    paddingHorizontal: SPACING.sm,
  },
  actionText: {
    ...TYPOGRAPHY.body,
    color: COLORS.primary,
    fontWeight: '600',
  },
});

