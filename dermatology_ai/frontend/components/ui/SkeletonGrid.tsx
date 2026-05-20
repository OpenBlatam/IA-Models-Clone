'use client';

import React, { memo, useMemo } from 'react';
import { SkeletonCard } from './Skeleton';
import { Grid } from './Grid';

interface SkeletonGridProps {
  count?: number;
  cols?: 1 | 2 | 3 | 4;
  gap?: 2 | 4 | 6 | 8;
  className?: string;
}

export const SkeletonGrid: React.FC<SkeletonGridProps> = memo(({
  count = 4,
  cols = 4,
  gap = 6,
  className,
}) => {
  const skeletonCards = useMemo(
    () => Array.from({ length: count }, (_, i) => <SkeletonCard key={i} />),
    [count]
  );

  return (
    <Grid cols={cols} gap={gap} className={className}>
      {skeletonCards}
    </Grid>
  );
});

SkeletonGrid.displayName = 'SkeletonGrid';

