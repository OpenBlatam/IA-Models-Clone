'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface TimelineItem {
  title: string;
  description?: string;
  time?: string;
  icon?: ReactNode;
  status?: 'completed' | 'current' | 'upcoming';
}

interface TimelineProps {
  items: TimelineItem[];
  className?: string;
}

const Timeline = ({ items, className }: TimelineProps) => {
  return (
    <div className={cn('relative', className)}>
      {items.map((item, index) => {
        const isLast = index === items.length - 1;
        const statusClasses = {
          completed: 'bg-green-500 border-green-500',
          current: 'bg-blue-500 border-blue-500 animate-pulse',
          upcoming: 'bg-gray-300 border-gray-300',
        };

        return (
          <div key={index} className="relative flex gap-4 pb-8">
            {!isLast && (
              <div className="absolute left-4 top-8 bottom-0 w-0.5 bg-gray-200" />
            )}
            <div
              className={cn(
                'relative z-10 flex h-8 w-8 items-center justify-center rounded-full border-2',
                statusClasses[item.status || 'upcoming']
              )}
            >
              {item.icon || (
                <div className={cn('h-2 w-2 rounded-full', item.status === 'completed' ? 'bg-white' : 'bg-transparent')} />
              )}
            </div>
            <div className="flex-1 pt-1">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-gray-900">{item.title}</h4>
                {item.time && <span className="text-xs text-gray-500">{item.time}</span>}
              </div>
              {item.description && (
                <p className="mt-1 text-sm text-gray-600">{item.description}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export { Timeline };

