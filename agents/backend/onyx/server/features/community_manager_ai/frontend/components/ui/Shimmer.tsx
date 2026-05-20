'use client';

import { cn } from '@/lib/utils';

interface ShimmerProps {
  className?: string;
  width?: string;
  height?: string;
}

export const Shimmer = ({ className, width = '100%', height = '100%' }: ShimmerProps) => {
  return (
    <div
      className={cn(
        'animate-shimmer bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200 dark:from-gray-800 dark:via-gray-700 dark:to-gray-800',
        'bg-[length:200%_100%]',
        className
      )}
      style={{ width, height }}
    />
  );
};



