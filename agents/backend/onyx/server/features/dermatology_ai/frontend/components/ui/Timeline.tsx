'use client';

import React from 'react';
import { clsx } from 'clsx';
import { CheckCircle, Circle } from 'lucide-react';

interface TimelineItem {
  id: string;
  title: string;
  description?: string;
  date?: string;
  status?: 'completed' | 'current' | 'pending';
}

interface TimelineProps {
  items: TimelineItem[];
  className?: string;
}

export const Timeline: React.FC<TimelineProps> = ({ items, className }) => {
  return (
    <div className={clsx('relative', className)}>
      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-700" />
      <div className="space-y-6">
        {items.map((item, index) => {
          const isCompleted = item.status === 'completed';
          const isCurrent = item.status === 'current';
          const isPending = item.status === 'pending' || !item.status;

          return (
            <div key={item.id} className="relative flex items-start space-x-4">
              <div
                className={clsx(
                  'relative z-10 flex items-center justify-center w-8 h-8 rounded-full border-2',
                  isCompleted &&
                    'bg-primary-600 border-primary-600 text-white',
                  isCurrent &&
                    'bg-primary-50 border-primary-600 text-primary-600 dark:bg-primary-900/20 dark:text-primary-400',
                  isPending &&
                    'bg-white border-gray-300 text-gray-400 dark:bg-gray-800 dark:border-gray-700'
                )}
              >
                {isCompleted ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <Circle className="h-5 w-5" />
                )}
              </div>
              <div className="flex-1 min-w-0 pb-6">
                <div className="flex items-center justify-between">
                  <h4
                    className={clsx(
                      'text-sm font-semibold',
                      isCompleted || isCurrent
                        ? 'text-gray-900 dark:text-white'
                        : 'text-gray-500 dark:text-gray-400'
                    )}
                  >
                    {item.title}
                  </h4>
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
            </div>
          );
        })}
      </div>
    </div>
  );
};


