'use client';

import { cn } from '@/lib/utils';

interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'away' | 'busy';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  className?: string;
}

const statusColors = {
  online: 'bg-green-500 dark:bg-green-400',
  offline: 'bg-gray-400 dark:bg-gray-500',
  away: 'bg-yellow-500 dark:bg-yellow-400',
  busy: 'bg-red-500 dark:bg-red-400',
};

const statusLabels = {
  online: 'En línea',
  offline: 'Desconectado',
  away: 'Ausente',
  busy: 'Ocupado',
};

const sizeClasses = {
  sm: 'h-2 w-2',
  md: 'h-3 w-3',
  lg: 'h-4 w-4',
};

export const StatusIndicator = ({
  status,
  size = 'md',
  showLabel = false,
  label,
  className,
}: StatusIndicatorProps) => {
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <span
        className={cn(
          'rounded-full',
          statusColors[status],
          sizeClasses[size],
          'ring-2 ring-white dark:ring-gray-800'
        )}
        aria-label={statusLabels[status]}
      />
      {showLabel && (
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {label || statusLabels[status]}
        </span>
      )}
    </div>
  );
};



