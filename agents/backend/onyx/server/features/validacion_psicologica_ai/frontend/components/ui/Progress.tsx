/**
 * Progress component (enhanced version)
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface ProgressProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  variant?: 'default' | 'success' | 'warning' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  striped?: boolean;
  className?: string;
}

const variantClasses = {
  default: 'bg-primary',
  success: 'bg-green-600',
  warning: 'bg-yellow-600',
  destructive: 'bg-red-600',
};

const sizeClasses = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-4',
};

export const Progress: React.FC<ProgressProps> = ({
  value,
  max = 100,
  label,
  showValue = false,
  variant = 'default',
  size = 'md',
  animated = false,
  striped = false,
  className,
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  return (
    <div className={cn('w-full', className)}>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1">
          {label && <span className="text-sm font-medium">{label}</span>}
          {showValue && (
            <span className="text-sm text-muted-foreground">{Math.round(percentage)}%</span>
          )}
        </div>
      )}
      <div
        className={cn(
          'w-full bg-muted rounded-full overflow-hidden',
          sizeClasses[size]
        )}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label || 'Progreso'}
      >
        <div
          className={cn(
            'h-full transition-all duration-300',
            variantClasses[variant],
            animated && 'animate-pulse',
            striped && 'bg-stripes bg-[length:1rem_1rem]'
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
