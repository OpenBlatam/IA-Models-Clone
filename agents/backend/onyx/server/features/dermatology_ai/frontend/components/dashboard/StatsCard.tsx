'use client';

import React, { memo } from 'react';
import { Card, CardContent } from '../ui/Card';
import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface StatsCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  subtitle?: string;
  className?: string;
  iconColor?: string;
}

export const StatsCard: React.FC<StatsCardProps> = memo(({
  title,
  value,
  icon: Icon,
  trend,
  subtitle,
  className,
  iconColor = 'text-primary-600',
}) => {
  return (
    <Card className={clsx('hover:shadow-lg transition-shadow', className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
              {title}
            </p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
              {value}
            </p>
            {subtitle && (
              <p className="text-xs text-gray-500 dark:text-gray-500">
                {subtitle}
              </p>
            )}
            {trend && (
              <div className="flex items-center mt-2">
                <span
                  className={clsx(
                    'text-sm font-medium',
                    trend.isPositive
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-red-600 dark:text-red-400'
                  )}
                >
                  {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">
                  vs previous
                </span>
              </div>
            )}
          </div>
          <div
            className={clsx(
              'p-3 rounded-lg bg-opacity-10',
              iconColor.includes('primary')
                ? 'bg-primary-100 dark:bg-primary-900'
                : iconColor.includes('green')
                ? 'bg-green-100 dark:bg-green-900'
                : iconColor.includes('yellow')
                ? 'bg-yellow-100 dark:bg-yellow-900'
                : 'bg-blue-100 dark:bg-blue-900'
            )}
          >
            <Icon className={clsx('h-6 w-6', iconColor)} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

StatsCard.displayName = 'StatsCard';

