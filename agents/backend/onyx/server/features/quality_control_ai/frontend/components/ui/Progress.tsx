'use client';

import { memo } from 'react';
import { cn } from '@/lib/utils';

interface ProgressProps {
  value: number;
  max?: number;
  className?: string;
  showLabel?: boolean;
  label?: string;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  striped?: boolean;
}

const Progress = memo(
  ({
    value,
    max = 100,
    className,
    showLabel = false,
    label,
    variant = 'default',
    size = 'md',
    animated = false,
    striped = false,
  }: ProgressProps): JSX.Element => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    const variantClasses = {
      default: 'bg-primary-600',
      success: 'bg-green-600',
      warning: 'bg-yellow-600',
      error: 'bg-red-600',
      info: 'bg-blue-600',
    };

    const sizeClasses = {
      sm: 'h-1',
      md: 'h-2',
      lg: 'h-3',
    };

    return (
      <div className={cn('w-full', className)} role="progressbar" aria-valuenow={value} aria-valuemin={0} aria-valuemax={max}>
        {(showLabel || label) && (
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>{label || 'Progress'}</span>
            <span>{Math.round(percentage)}%</span>
          </div>
        )}
        <div className={cn('w-full bg-gray-200 rounded-full overflow-hidden', sizeClasses[size])}>
          <div
            className={cn(
              variantClasses[variant],
              'h-full transition-all duration-300 ease-in-out',
              animated && 'animate-pulse',
              striped && 'bg-stripes'
            )}
            style={{
              width: `${percentage}%`,
              ...(striped && {
                backgroundImage: 'linear-gradient(45deg, rgba(255,255,255,.15) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.15) 50%, rgba(255,255,255,.15) 75%, transparent 75%, transparent)',
                backgroundSize: '1rem 1rem',
              }),
            }}
            aria-label={`Progress: ${Math.round(percentage)}%`}
          />
        </div>
      </div>
    );
  }
);

Progress.displayName = 'Progress';

export default Progress;

