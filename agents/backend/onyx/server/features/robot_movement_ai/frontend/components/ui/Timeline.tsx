'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface TimelineItem {
  id: string;
  title: string;
  description?: string;
  date?: string;
  icon?: ReactNode;
  status?: 'completed' | 'active' | 'pending';
  color?: string;
}

interface TimelineProps {
  items: TimelineItem[];
  orientation?: 'vertical' | 'horizontal';
  className?: string;
}

export default function Timeline({ items, orientation = 'vertical', className }: TimelineProps) {
  if (orientation === 'horizontal') {
    return (
      <div className={cn('flex items-center gap-4 overflow-x-auto pb-4', className)}>
        {items.map((item, index) => (
          <div key={item.id} className="flex items-center flex-shrink-0">
            <TimelineItemHorizontal item={item} index={index} />
            {index < items.length - 1 && (
              <div className="w-16 h-0.5 bg-gray-300 mx-2" />
            )}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={cn('relative', className)}>
      {/* Vertical Line */}
      <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-200" />

      {items.map((item, index) => (
        <TimelineItemVertical key={item.id} item={item} index={index} />
      ))}
    </div>
  );
}

function TimelineItemVertical({ item, index }: { item: TimelineItem; index: number }) {
  const statusColors = {
    completed: 'bg-green-500',
    active: 'bg-tesla-blue',
    pending: 'bg-gray-300',
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
      className="relative flex gap-4 pb-8 last:pb-0"
    >
      {/* Icon/Status Circle */}
      <div className="relative z-10 flex-shrink-0">
        <div
          className={cn(
            'w-12 h-12 rounded-full flex items-center justify-center border-4 border-white shadow-sm',
            statusColors[item.status || 'pending']
          )}
        >
          {item.icon || (
            <div className="w-2 h-2 rounded-full bg-white" />
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 pt-1">
        {item.date && (
          <div className="text-xs text-tesla-gray-dark mb-1">{item.date}</div>
        )}
        <h3 className="text-base font-semibold text-tesla-black mb-1">{item.title}</h3>
        {item.description && (
          <p className="text-sm text-tesla-gray-dark">{item.description}</p>
        )}
      </div>
    </motion.div>
  );
}

function TimelineItemHorizontal({ item, index }: { item: TimelineItem; index: number }) {
  const statusColors = {
    completed: 'bg-green-500',
    active: 'bg-tesla-blue',
    pending: 'bg-gray-300',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
      className="flex flex-col items-center text-center min-w-[120px]"
    >
      <div
        className={cn(
          'w-10 h-10 rounded-full flex items-center justify-center border-4 border-white shadow-sm mb-2',
          statusColors[item.status || 'pending']
        )}
      >
        {item.icon || (
          <div className="w-2 h-2 rounded-full bg-white" />
        )}
      </div>
      {item.date && (
        <div className="text-xs text-tesla-gray-dark mb-1">{item.date}</div>
      )}
      <h3 className="text-sm font-semibold text-tesla-black">{item.title}</h3>
      {item.description && (
        <p className="text-xs text-tesla-gray-dark mt-1">{item.description}</p>
      )}
    </motion.div>
  );
}



