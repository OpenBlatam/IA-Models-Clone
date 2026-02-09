'use client';

import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface Stat {
  label: string;
  value: string | number;
  icon?: ReactNode;
  change?: {
    value: string;
    positive: boolean;
  };
}

interface StatsGridProps {
  stats: Stat[];
  className?: string;
  columns?: 2 | 3 | 4;
}

export default function StatsGrid({ stats, className, columns = 4 }: StatsGridProps) {
  const gridCols = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-3',
    4: 'grid-cols-2 md:grid-cols-4',
  };

  return (
    <div className={cn('grid gap-4 md:gap-6', gridCols[columns], className)}>
      {stats.map((stat, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
          className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm hover:shadow-tesla-md transition-all"
        >
          <div className="flex items-start justify-between mb-2">
            {stat.icon && <div className="text-tesla-blue">{stat.icon}</div>}
            {stat.change && (
              <span
                className={cn(
                  'text-sm font-medium',
                  stat.change.positive ? 'text-green-600' : 'text-red-600'
                )}
              >
                {stat.change.positive ? '+' : ''}
                {stat.change.value}
              </span>
            )}
          </div>
          <p className="text-2xl md:text-3xl font-bold text-tesla-black mb-1">{stat.value}</p>
          <p className="text-sm text-tesla-gray-dark">{stat.label}</p>
        </motion.div>
      ))}
    </div>
  );
}



