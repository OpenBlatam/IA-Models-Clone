'use client';

import React from 'react';
import { clsx } from 'clsx';
import { User } from 'lucide-react';

interface AvatarProps {
  src?: string;
  alt?: string;
  name?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  status?: 'online' | 'offline' | 'away';
}

export const Avatar: React.FC<AvatarProps> = ({
  src,
  alt,
  name,
  size = 'md',
  className,
  status,
}) => {
  const sizes = {
    sm: 'h-8 w-8 text-xs',
    md: 'h-10 w-10 text-sm',
    lg: 'h-12 w-12 text-base',
    xl: 'h-16 w-16 text-lg',
  };

  const statusSizes = {
    sm: 'h-2 w-2',
    md: 'h-2.5 w-2.5',
    lg: 'h-3 w-3',
    xl: 'h-3.5 w-3.5',
  };

  const statusColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-400',
    away: 'bg-yellow-500',
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className={clsx('relative inline-block', className)}>
      {src ? (
        <img
          src={src}
          alt={alt || name}
          className={clsx(
            'rounded-full object-cover',
            sizes[size]
          )}
        />
      ) : (
        <div
          className={clsx(
            'rounded-full bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 flex items-center justify-center font-semibold',
            sizes[size]
          )}
        >
          {name ? getInitials(name) : <User className="h-1/2 w-1/2" />}
        </div>
      )}
      {status && (
        <span
          className={clsx(
            'absolute bottom-0 right-0 rounded-full border-2 border-white dark:border-gray-800',
            statusSizes[size],
            statusColors[status]
          )}
        />
      )}
    </div>
  );
};


