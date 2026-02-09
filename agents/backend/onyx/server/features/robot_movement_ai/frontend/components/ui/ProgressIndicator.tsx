'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface ProgressIndicatorProps {
  value: number;
  max?: number;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'green' | 'yellow' | 'red';
  className?: string;
  animated?: boolean;
}

const sizeClasses = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-3',
};

const colorClasses = {
  blue: 'bg-tesla-blue',
  green: 'bg-[#10b981]',
  yellow: 'bg-[#f59e0b]',
  red: 'bg-[#ef4444]',
};

export default function ProgressIndicator({
  value,
  max = 100,
  showLabel = false,
  size = 'md',
  color = 'blue',
  className,
  animated = true,
}: ProgressIndicatorProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  return (
    <div className={cn('w-full', className)}>
      {showLabel && (
        <div className="flex items-center justify-between mb-tesla-sm">
          <span className="text-sm font-medium text-tesla-black">
            {value} / {max}
          </span>
          <span className="text-sm text-tesla-gray-dark">{Math.round(percentage)}%</span>
        </div>
      )}
      <div
        className={cn(
          'w-full rounded-full overflow-hidden',
          'bg-[#e5e7eb]',
          sizeClasses[size]
        )}
      >
        <motion.div
          initial={animated ? { width: 0 } : { width: `${percentage}%` }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className={cn('h-full rounded-full', colorClasses[color])}
        />
      </div>
    </div>
  );
}

