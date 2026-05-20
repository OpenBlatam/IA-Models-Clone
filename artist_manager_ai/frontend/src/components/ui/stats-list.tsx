'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface StatItem {
  label: string;
  value: string | number;
  icon?: ReactNode;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
}

interface StatsListProps {
  items: StatItem[];
  className?: string;
  showDividers?: boolean;
}

const variantColors = {
  default: 'text-gray-900',
  primary: 'text-blue-600',
  success: 'text-green-600',
  warning: 'text-yellow-600',
  danger: 'text-red-600',
};

const StatsList = ({ items, className, showDividers = true }: StatsListProps) => {
  return (
    <div className={cn('space-y-4', className)}>
      {items.map((item, index) => (
        <div
          key={index}
          className={cn(
            'flex justify-between items-center py-2',
            showDividers && index < items.length - 1 && 'border-b'
          )}
        >
          <div className="flex items-center gap-2">
            {item.icon && <div className="text-gray-400">{item.icon}</div>}
            <span className="text-gray-600">{item.label}</span>
          </div>
          <span className={cn('font-semibold text-lg', variantColors[item.variant || 'default'])}>
            {item.value}
          </span>
        </div>
      ))}
    </div>
  );
};

export { StatsList };

