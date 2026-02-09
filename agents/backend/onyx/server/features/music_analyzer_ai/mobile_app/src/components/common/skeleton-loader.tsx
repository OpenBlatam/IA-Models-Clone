import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';

interface SkeletonLoaderProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  style?: unknown;
}

export function SkeletonLoader({
  width = '100%',
  height = 20,
  borderRadius = BORDER_RADIUS.sm,
  style,
}: SkeletonLoaderProps) {
  const opacity = useSharedValue(0.3);

  useEffect(() => {
    opacity.value = withRepeat(
      withTiming(1, { duration: 1000 }),
      -1,
      true
    );
  }, [opacity]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: interpolate(opacity.value, [0.3, 1], [0.3, 0.7]),
  }));

  return (
    <Animated.View
      style={[
        {
          width,
          height,
          borderRadius,
          backgroundColor: COLORS.surfaceLight,
        },
        animatedStyle,
        style,
      ]}
    />
  );
}

interface SkeletonCardProps {
  showImage?: boolean;
}

export function SkeletonCard({ showImage = true }: SkeletonCardProps) {
  return (
    <View style={styles.card}>
      {showImage && (
        <SkeletonLoader
          width={60}
          height={60}
          borderRadius={BORDER_RADIUS.sm}
          style={styles.image}
        />
      )}
      <View style={styles.content}>
        <SkeletonLoader width="80%" height={16} style={styles.title} />
        <SkeletonLoader width="60%" height={14} style={styles.subtitle} />
        <SkeletonLoader width="40%" height={12} style={styles.footer} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: BORDER_RADIUS.md,
    padding: SPACING.md,
    marginBottom: SPACING.sm,
  },
  image: {
    marginRight: SPACING.md,
  },
  content: {
    flex: 1,
    justifyContent: 'space-between',
  },
  title: {
    marginBottom: SPACING.xs,
  },
  subtitle: {
    marginBottom: SPACING.xs,
  },
  footer: {
    marginTop: SPACING.xs,
  },
});

