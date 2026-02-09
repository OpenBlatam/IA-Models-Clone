'use client';

import { motion } from 'framer-motion';
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import AnimatedNumber from './AnimatedNumber';

interface StatCardProps {
  title: string;
  value: number | string;
  change?: {
    value: number;
    label?: string;
  };
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  format?: (value: number) => string;
  className?: string;
  highlight?: boolean;
}

export default function StatCard({
  title,
  value,
  change,
  icon: Icon,
  trend,
  format,
  className,
  highlight = false,
}: StatCardProps) {
  const isNumber = typeof value === 'number';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      whileHover={{ y: -4, scale: 1.02 }}
      className={cn(
        'bg-white rounded-lg border border-gray-200 p-6 shadow-sm transition-all',
        'hover:shadow-tesla-md hover:border-gray-300',
        highlight && 'ring-2 ring-tesla-blue/20',
        className
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <p className="text-sm font-medium text-tesla-gray-dark mb-1">{title}</p>
          <div className="flex items-baseline gap-2">
            {isNumber ? (
              <AnimatedNumber
                value={value}
                format={format}
                className="text-2xl md:text-3xl font-bold text-tesla-black"
              />
            ) : (
              <p className="text-2xl md:text-3xl font-bold text-tesla-black">{value}</p>
            )}
          </div>
        </div>
        {Icon && (
          <div className="p-3 bg-tesla-blue/10 rounded-lg">
            <Icon className="w-5 h-5 text-tesla-blue" />
          </div>
        )}
      </div>

      {change && (
        <div className="flex items-center gap-2">
          {trend === 'up' ? (
            <TrendingUp className="w-4 h-4 text-green-600" />
          ) : trend === 'down' ? (
            <TrendingDown className="w-4 h-4 text-red-600" />
          ) : null}
          <span
            className={cn(
              'text-sm font-medium',
              trend === 'up' && 'text-green-600',
              trend === 'down' && 'text-red-600',
              trend === 'neutral' && 'text-tesla-gray-dark'
            )}
          >
            {change.value > 0 ? '+' : ''}
            {change.value}% {change.label}
          </span>
        </div>
      )}
    </motion.div>
  );
}



