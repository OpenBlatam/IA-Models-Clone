'use client';

import { memo } from 'react';
import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

const Skeleton = memo(
  ({
    className,
    variant = 'rectangular',
    width,
    height,
    animation = 'pulse',
  }: SkeletonProps): JSX.Element => {
    const variantClasses = {
      text: 'h-4 rounded',
      circular: 'rounded-full',
      rectangular: 'rounded',
    };

    const animationClasses = {
      pulse: 'animate-pulse',
      wave: 'animate-pulse',
      none: '',
    };

    const style: React.CSSProperties = {
      ...(width && { width: typeof width === 'number' ? `${width}px` : width }),
      ...(height && { height: typeof height === 'number' ? `${height}px` : height }),
    };

    return (
      <div
        className={cn(
          'bg-gray-200',
          variantClasses[variant],
          animationClasses[animation],
          className
        )}
        style={style}
        aria-hidden="true"
        role="status"
        aria-label="Loading"
      />
    );
  }
);

Skeleton.displayName = 'Skeleton';

export default Skeleton;

