'use client';

import { Skeleton } from './Skeleton';
import { cn } from '@/lib/utils/cn';

interface SkeletonLoaderProps {
  variant?: 'card' | 'list' | 'table' | 'profile' | 'product';
  count?: number;
  className?: string;
}

export default function SkeletonLoader({
  variant = 'card',
  count = 1,
  className,
}: SkeletonLoaderProps) {
  if (variant === 'card') {
    return (
      <div className={cn('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', className)}>
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="bg-white rounded-lg border border-gray-200 p-6">
            <Skeleton variant="rectangular" className="w-full h-48 mb-4" />
            <Skeleton variant="text" className="w-3/4 h-6 mb-2" />
            <Skeleton variant="text" className="w-full h-4 mb-2" />
            <Skeleton variant="text" className="w-2/3 h-4" />
          </div>
        ))}
      </div>
    );
  }

  if (variant === 'list') {
    return (
      <div className={cn('space-y-4', className)}>
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="flex items-center gap-4">
            <Skeleton variant="circular" className="w-12 h-12" />
            <div className="flex-1 space-y-2">
              <Skeleton variant="text" className="w-1/4 h-4" />
              <Skeleton variant="text" className="w-1/2 h-3" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (variant === 'table') {
    return (
      <div className={cn('space-y-3', className)}>
        {/* Header */}
        <div className="flex gap-4 pb-3 border-b border-gray-200">
          <Skeleton variant="text" className="w-1/4 h-4" />
          <Skeleton variant="text" className="w-1/4 h-4" />
          <Skeleton variant="text" className="w-1/4 h-4" />
          <Skeleton variant="text" className="w-1/4 h-4" />
        </div>
        {/* Rows */}
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="flex gap-4 py-3">
            <Skeleton variant="text" className="w-1/4 h-4" />
            <Skeleton variant="text" className="w-1/4 h-4" />
            <Skeleton variant="text" className="w-1/4 h-4" />
            <Skeleton variant="text" className="w-1/4 h-4" />
          </div>
        ))}
      </div>
    );
  }

  if (variant === 'profile') {
    return (
      <div className={cn('space-y-6', className)}>
        <div className="flex items-center gap-6">
          <Skeleton variant="circular" className="w-24 h-24" />
          <div className="flex-1 space-y-3">
            <Skeleton variant="text" className="w-1/3 h-6" />
            <Skeleton variant="text" className="w-1/2 h-4" />
            <Skeleton variant="text" className="w-1/4 h-4" />
          </div>
        </div>
        <div className="space-y-4">
          <Skeleton variant="text" className="w-full h-4" />
          <Skeleton variant="text" className="w-full h-4" />
          <Skeleton variant="text" className="w-3/4 h-4" />
        </div>
      </div>
    );
  }

  if (variant === 'product') {
    return (
      <div className={cn('space-y-4', className)}>
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="flex gap-4">
            <Skeleton variant="rectangular" className="w-24 h-24 flex-shrink-0" />
            <div className="flex-1 space-y-2">
              <Skeleton variant="text" className="w-1/2 h-5" />
              <Skeleton variant="text" className="w-1/3 h-4" />
              <Skeleton variant="text" className="w-1/4 h-4" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return null;
}



