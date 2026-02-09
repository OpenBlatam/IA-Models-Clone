'use client';

import { memo, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  label: string;
  value: ReactNode;
  className?: string;
  valueClassName?: string;
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  'aria-label'?: string;
}

const StatCard = memo(
  ({
    label,
    value,
    className,
    valueClassName,
    icon,
    trend,
    trendValue,
    'aria-label': ariaLabel,
  }: StatCardProps): JSX.Element => {
    const trendColors = {
      up: 'text-green-600',
      down: 'text-red-600',
      neutral: 'text-gray-600',
    };

    return (
      <div
        className={cn('bg-gray-50 rounded-lg p-4 transition-shadow hover:shadow-sm', className)}
        role="region"
        aria-label={ariaLabel || `${label}: ${value}`}
      >
        <div className="flex items-center justify-between mb-1">
          <div className="text-sm text-gray-600">{label}</div>
          {icon && <div className="text-gray-400" aria-hidden="true">{icon}</div>}
        </div>
        <div className={cn('text-2xl font-bold text-gray-900', valueClassName)}>
          {value}
        </div>
        {trend && trendValue && (
          <div className={cn('text-xs mt-1', trendColors[trend])} aria-label={`Trend: ${trend}, ${trendValue}`}>
            {trend === 'up' && '↑'} {trend === 'down' && '↓'} {trendValue}
          </div>
        )}
      </div>
    );
  }
);

StatCard.displayName = 'StatCard';

export default StatCard;

