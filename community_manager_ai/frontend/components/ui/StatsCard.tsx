'use client';

import { Card, CardContent } from './Card';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';
import { motion } from 'framer-motion';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: {
    value: number;
    label: string;
    positive?: boolean;
  };
  icon?: LucideIcon;
  iconColor?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export const StatsCard = ({
  title,
  value,
  change,
  icon: Icon,
  iconColor,
  trend = 'neutral',
  className,
}: StatsCardProps) => {
  const trendColors = {
    up: 'text-green-600 dark:text-green-400',
    down: 'text-red-600 dark:text-red-400',
    neutral: 'text-gray-600 dark:text-gray-400',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={cn('hover:shadow-lg transition-shadow', className)}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
              <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-gray-100">{value}</p>
              {change && (
                <div className="mt-2 flex items-center gap-1">
                  <span
                    className={cn(
                      'text-sm font-medium',
                      change.positive !== undefined
                        ? change.positive
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                        : trendColors[trend]
                    )}
                  >
                    {change.positive !== undefined && (change.positive ? '+' : '-')}
                    {Math.abs(change.value)}%
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-500">{change.label}</span>
                </div>
              )}
            </div>
            {Icon && (
              <div
                className={cn(
                  'flex h-12 w-12 items-center justify-center rounded-lg',
                  iconColor || 'bg-primary-100 dark:bg-primary-900/20'
                )}
              >
                <Icon
                  className={cn(
                    'h-6 w-6',
                    iconColor || 'text-primary-600 dark:text-primary-400'
                  )}
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};



