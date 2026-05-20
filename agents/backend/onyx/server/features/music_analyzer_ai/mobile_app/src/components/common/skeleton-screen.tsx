import React, { memo } from 'react';
import { View, StyleSheet } from 'react-native';
import { COLORS, SPACING, BORDER_RADIUS } from '../../constants/config';
import { SkeletonLoader } from './skeleton-loader';

interface SkeletonScreenProps {
  itemCount?: number;
  showHeader?: boolean;
  showImage?: boolean;
  itemHeight?: number;
}

function SkeletonScreenComponent({
  itemCount = 5,
  showHeader = true,
  showImage = true,
  itemHeight = 80,
}: SkeletonScreenProps) {
  return (
    <View style={styles.container}>
      {showHeader && (
        <View style={styles.header}>
          <SkeletonLoader width={200} height={24} borderRadius={BORDER_RADIUS.sm} />
          <SkeletonLoader width={150} height={16} borderRadius={BORDER_RADIUS.sm} />
        </View>
      )}
      {Array.from({ length: itemCount }).map((_, index) => (
        <View key={index} style={[styles.item, { height: itemHeight }]}>
          {showImage && (
            <SkeletonLoader
              width={60}
              height={60}
              borderRadius={BORDER_RADIUS.sm}
              style={styles.image}
            />
          )}
          <View style={styles.content}>
            <SkeletonLoader width="70%" height={16} borderRadius={BORDER_RADIUS.sm} />
            <SkeletonLoader
              width="50%"
              height={14}
              borderRadius={BORDER_RADIUS.sm}
              style={styles.secondLine}
            />
            <SkeletonLoader
              width="40%"
              height={12}
              borderRadius={BORDER_RADIUS.sm}
              style={styles.thirdLine}
            />
          </View>
        </View>
      ))}
    </View>
  );
}

export const SkeletonScreen = memo(SkeletonScreenComponent);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: SPACING.md,
    backgroundColor: COLORS.background,
  },
  header: {
    marginBottom: SPACING.lg,
    gap: SPACING.sm,
  },
  item: {
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
  secondLine: {
    marginTop: SPACING.xs,
  },
  thirdLine: {
    marginTop: SPACING.xs,
  },
});
