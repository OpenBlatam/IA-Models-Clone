'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface TimelineItem {
  id: string;
  title: string;
  description?: string;
  date?: string;
  icon?: ReactNode;
  color?: 'primary' | 'success' | 'warning' | 'error';
}

interface TimelineProps {
  items: TimelineItem[];
  className?: string;
}

const colorClasses = {
  primary: 'bg-primary-600 dark:bg-primary-500',
  success: 'bg-green-600 dark:bg-green-500',
  warning: 'bg-yellow-600 dark:bg-yellow-500',
  error: 'bg-red-600 dark:bg-red-500',
};

export const Timeline = ({ items, className }: TimelineProps) => {
  return (
    <div className={cn('relative', className)}>
      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700" />
      <div className="space-y-6">
        {items.map((item, index) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative flex items-start gap-4"
          >
            <div
              className={cn(
                'relative z-10 flex h-8 w-8 items-center justify-center rounded-full border-2 border-white dark:border-gray-800',
                colorClasses[item.color || 'primary']
              )}
            >
              {item.icon || (
                <div className="h-2 w-2 rounded-full bg-white dark:bg-gray-800" />
              )}
            </div>
            <div className="flex-1 pb-6">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                  {item.title}
                </h3>
                {item.date && (
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {item.date}
                  </span>
                )}
              </div>
              {item.description && (
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                  {item.description}
                </p>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};



