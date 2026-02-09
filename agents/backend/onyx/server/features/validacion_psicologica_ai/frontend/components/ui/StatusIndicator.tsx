/**
 * Status indicator component
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'away' | 'busy';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  pulse?: boolean;
}

const statusColors = {
  online: 'bg-green-500',
  offline: 'bg-gray-400',
  away: 'bg-yellow-500',
  busy: 'bg-red-500',
};

const sizeClasses = {
  sm: 'h-2 w-2',
  md: 'h-3 w-3',
  lg: 'h-4 w-4',
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  size = 'md',
  className,
  pulse = false,
}) => {
  return (
    <div className={cn('relative inline-flex items-center', className)}>
      <div
        className={cn(
          'rounded-full',
          statusColors[status],
          sizeClasses[size]
        )}
        aria-label={`Estado: ${status}`}
      />
      {pulse && status === 'online' && (
        <div
          className={cn(
            'absolute inset-0 rounded-full animate-ping',
            statusColors[status],
            sizeClasses[size]
          )}
          aria-hidden="true"
        />
      )}
    </div>
  );
};



