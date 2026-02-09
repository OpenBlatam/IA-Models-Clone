import React from 'react';
import { View, StyleSheet, ActivityIndicator, Text } from 'react-native';
import { BlurView } from 'expo-blur';
import { COLORS, SPACING, TYPOGRAPHY } from '../../constants/config';

interface LoadingOverlayProps {
  visible: boolean;
  message?: string;
  transparent?: boolean;
}

/**
 * Full-screen loading overlay with blur effect
 * Use for blocking operations
 */
export function LoadingOverlay({
  visible,
  message,
  transparent = false,
}: LoadingOverlayProps) {
  if (!visible) return null;

  const content = (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={COLORS.primary} />
      {message && <Text style={styles.message}>{message}</Text>}
    </View>
  );

  if (transparent) {
    return (
      <View style={styles.overlay}>
        <BlurView intensity={20} style={styles.blur}>
          {content}
        </BlurView>
      </View>
    );
  }

  return <View style={[styles.overlay, styles.solid]}>{content}</View>;
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 9999,
    justifyContent: 'center',
    alignItems: 'center',
  },
  solid: {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  blur: {
    ...StyleSheet.absoluteFillObject,
  },
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: SPACING.lg,
  },
  message: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
    marginTop: SPACING.md,
    textAlign: 'center',
  },
});

