/**
 * Skeleton Component
 * =================
 * Loading skeleton component
 */

import { View, StyleSheet } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
  interpolate,
} from 'react-native-reanimated';
import { useEffect } from 'react';
import { useApp } from '@/lib/context/app-context';

interface SkeletonProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  style?: any;
}

export function Skeleton({ width = '100%', height = 20, borderRadius = 4, style }: SkeletonProps) {
  const { state } = useApp();
  const colors = state.colors;
  const opacity = useSharedValue(0.3);

  useEffect(() => {
    opacity.value = withRepeat(
      withTiming(1, { duration: 1000 }),
      -1,
      true
    );
  }, []);

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
          backgroundColor: colors.backgroundSecondary,
        },
        animatedStyle,
        style,
      ]}
    />
  );
}

export function SkeletonText({ lines = 3, style }: { lines?: number; style?: any }) {
  return (
    <View style={style}>
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton
          key={index}
          height={16}
          width={index === lines - 1 ? '80%' : '100%'}
          style={{ marginBottom: 8 }}
        />
      ))}
    </View>
  );
}

export function SkeletonCard() {
  return (
    <View style={styles.card}>
      <Skeleton width={60} height={60} borderRadius={30} style={styles.avatar} />
      <View style={styles.content}>
        <Skeleton width="70%" height={16} style={styles.title} />
        <Skeleton width="100%" height={14} style={styles.line} />
        <Skeleton width="90%" height={14} style={styles.line} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  avatar: {
    marginBottom: 0,
  },
  content: {
    flex: 1,
  },
  title: {
    marginBottom: 8,
  },
  line: {
    marginBottom: 6,
  },
});



