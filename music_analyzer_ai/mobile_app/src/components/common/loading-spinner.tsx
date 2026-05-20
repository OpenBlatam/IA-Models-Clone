import React, { memo } from 'react';
import { View, ActivityIndicator, StyleSheet, Text } from 'react-native';
import Animated, { FadeIn, FadeOut } from 'react-native-reanimated';
import { COLORS, TYPOGRAPHY, SPACING } from '../../constants/config';

interface LoadingSpinnerProps {
  size?: 'small' | 'large';
  color?: string;
  message?: string;
}

const AnimatedView = Animated.createAnimatedComponent(View);

export const LoadingSpinner = memo(function LoadingSpinner({
  size = 'large',
  color = COLORS.primary,
  message,
}: LoadingSpinnerProps) {
  return (
    <AnimatedView
      entering={FadeIn.duration(200)}
      exiting={FadeOut.duration(200)}
      style={styles.container}
    >
      <ActivityIndicator size={size} color={color} />
      {message && <Text style={styles.message}>{message}</Text>}
    </AnimatedView>
  );
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: SPACING.md,
  },
  message: {
    ...TYPOGRAPHY.body,
    color: COLORS.textSecondary,
    marginTop: SPACING.sm,
  },
});

