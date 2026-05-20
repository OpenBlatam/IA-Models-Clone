'use client';

import React from 'react';
import { clsx } from 'clsx';

interface StatusIndicatorProps {
  status: 'online' | 'offline' | 'away' | 'busy' | 'loading';
  size?: 'sm' | 'md' | 'lg';
  showPulse?: boolean;
  className?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  size = 'md',
  showPulse = true,
  className,
}) => {
  const sizes = {
    sm: 'h-2 w-2',
    md: 'h-3 w-3',
    lg: 'h-4 w-4',
  };

  const statusColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-400',
    away: 'bg-yellow-500',
    busy: 'bg-red-500',
    loading: 'bg-blue-500',
  };

  return (
    <div className={clsx('relative inline-flex items-center', className)}>
      <div
        className={clsx(
          'rounded-full',
          sizes[size],
          statusColors[status]
        )}
      />
      {showPulse && status === 'online' && (
        <div
          className={clsx(
            'absolute rounded-full animate-ping',
            sizes[size],
            statusColors[status],
            'opacity-75'
          )}
        />
      )}
    </div>
  );
};


