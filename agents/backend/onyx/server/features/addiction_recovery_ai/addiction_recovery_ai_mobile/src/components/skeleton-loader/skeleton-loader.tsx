import React from 'react';
import SkeletonPlaceholder from 'react-native-skeleton-placeholder';
import { useColors } from '@/theme/colors';

interface SkeletonLoaderProps {
  width?: number | string;
  height?: number;
  borderRadius?: number;
  children?: React.ReactNode;
}

export function SkeletonLoader({
  width = '100%',
  height = 20,
  borderRadius = 4,
  children,
}: SkeletonLoaderProps): JSX.Element {
  const colors = useColors();

  if (children) {
    return (
      <SkeletonPlaceholder
        backgroundColor={colors.border}
        highlightColor={colors.surface}
      >
        {children}
      </SkeletonPlaceholder>
    );
  }

  return (
    <SkeletonPlaceholder
      backgroundColor={colors.border}
      highlightColor={colors.surface}
    >
      <SkeletonPlaceholder.Item width={width} height={height} borderRadius={borderRadius} />
    </SkeletonPlaceholder>
  );
}

