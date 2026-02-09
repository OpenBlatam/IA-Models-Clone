'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';
import { LucideIcon } from 'lucide-react';

interface StatItemProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  iconColor?: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  subtitle?: string;
  className?: string;
}

export const StatItem: React.FC<StatItemProps> = memo(({
  label,
  value,
  icon: Icon,
  iconColor = 'text-primary-600 dark:text-primary-400',
  trend,
  subtitle,
  className,
}) => {
  return (
    <div className={clsx('flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg', className)}>
      <div className="flex items-center space-x-4 flex-1">
        {Icon && (
          <div className={clsx('p-2 rounded-lg bg-opacity-10', iconColor.includes('primary') ? 'bg-primary-100 dark:bg-primary-900' : 'bg-gray-100 dark:bg-gray-700')}>
            <Icon className={clsx('h-5 w-5', iconColor)} />
          </div>
        )}
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">{subtitle}</p>
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
              <span className="text-xs text-gray-500 dark:text-gray-400 ml-1">vs previous</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

StatItem.displayName = 'StatItem';



