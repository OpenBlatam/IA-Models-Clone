'use client';

import { cn } from '@/lib/utils';

interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'away' | 'busy';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-2 h-2',
  md: 'w-3 h-3',
  lg: 'w-4 h-4',
};

const statusClasses = {
  online: 'bg-green-500',
  offline: 'bg-gray-400',
  away: 'bg-yellow-500',
  busy: 'bg-red-500',
};

const StatusIndicator = ({ status, size = 'md', className }: StatusIndicatorProps) => {
  return (
    <span
      className={cn(
        'inline-block rounded-full',
        sizeClasses[size],
        statusClasses[status],
        className
      )}
      aria-label={`Estado: ${status}`}
      role="status"
    />
  );
};

export { StatusIndicator };

