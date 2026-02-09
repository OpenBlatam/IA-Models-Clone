'use client';

import React from 'react';
import { clsx } from 'clsx';

interface TimelineItem {
  id: string;
  title: string;
  description?: string;
  date?: string;
  icon?: React.ReactNode;
  status?: 'completed' | 'current' | 'upcoming';
}

interface TimelineVerticalProps {
  items: TimelineItem[];
  className?: string;
}

export const TimelineVertical: React.FC<TimelineVerticalProps> = ({
  items,
  className,
}) => {
  return (
    <div className={clsx('relative', className)}>
      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700" />
      <div className="space-y-8">
        {items.map((item, index) => {
          const isCompleted = item.status === 'completed';
          const isCurrent = item.status === 'current';

          return (
            <div key={item.id} className="relative flex items-start">
              <div
                className={clsx(
                  'relative z-10 flex items-center justify-center w-8 h-8 rounded-full border-2',
                  isCompleted
                    ? 'bg-primary-600 border-primary-600 text-white'
                    : isCurrent
                    ? 'bg-white dark:bg-gray-800 border-primary-600 text-primary-600'
                    : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-400'
                )}
              >
                {item.icon || (
                  <div
                    className={clsx(
                      'w-2 h-2 rounded-full',
                      isCompleted
                        ? 'bg-white'
                        : isCurrent
                        ? 'bg-primary-600'
                        : 'bg-gray-400'
                    )}
                  />
                )}
              </div>
              <div className="ml-4 flex-1">
                <div className="flex items-center justify-between">
                  <h4
                    className={clsx(
                      'font-medium',
                      isCompleted || isCurrent
                        ? 'text-gray-900 dark:text-white'
                        : 'text-gray-500 dark:text-gray-400'
                    )}
                  >
                    {item.title}
                  </h4>
                  {item.date && (
                    <span className="text-sm text-gray-500 dark:text-gray-400">
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
            </div>
          );
        })}
      </div>
    </div>
  );
};


