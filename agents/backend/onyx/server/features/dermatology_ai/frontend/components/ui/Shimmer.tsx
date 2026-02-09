'use client';

import React from 'react';
import { clsx } from 'clsx';

interface ShimmerProps {
  className?: string;
  width?: string | number;
  height?: string | number;
}

export const Shimmer: React.FC<ShimmerProps> = ({
  className,
  width = '100%',
  height = '1rem',
}) => {
  return (
    <div
      className={clsx(
        'bg-gray-200 dark:bg-gray-700 rounded animate-shimmer',
        className
      )}
      style={{
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
      }}
    />
  );
};


