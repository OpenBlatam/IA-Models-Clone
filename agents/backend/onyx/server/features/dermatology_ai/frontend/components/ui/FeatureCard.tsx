'use client';

import React, { memo } from 'react';
import { Card, CardContent } from './Card';
import { LucideIcon } from 'lucide-react';
import { clsx } from 'clsx';

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  className?: string;
  iconColor?: string;
  hover?: boolean;
}

export const FeatureCard: React.FC<FeatureCardProps> = memo(({
  icon: Icon,
  title,
  description,
  className,
  iconColor = 'text-primary-600 dark:text-primary-400',
  hover = true,
}) => {
  return (
    <Card
      className={clsx(
        'transition-all duration-300',
        hover && 'hover:shadow-lg hover:scale-105',
        className
      )}
    >
      <CardContent className="p-6">
        <div className="flex flex-col items-center text-center">
          <div
            className={clsx(
              'p-4 rounded-full bg-opacity-10 mb-4',
              iconColor.includes('primary')
                ? 'bg-primary-100 dark:bg-primary-900'
                : iconColor.includes('green')
                ? 'bg-green-100 dark:bg-green-900'
                : iconColor.includes('yellow')
                ? 'bg-yellow-100 dark:bg-yellow-900'
                : 'bg-blue-100 dark:bg-blue-900'
            )}
          >
            <Icon className={clsx('h-8 w-8', iconColor)} />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            {title}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {description}
          </p>
        </div>
      </CardContent>
    </Card>
  );
});

FeatureCard.displayName = 'FeatureCard';

