'use client';

import { cn } from '@/lib/utils';
import { User } from 'lucide-react';

interface AvatarProps {
  src?: string;
  alt?: string;
  name?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

const sizeClasses = {
  sm: 'w-8 h-8 text-xs',
  md: 'w-10 h-10 text-sm',
  lg: 'w-12 h-12 text-base',
  xl: 'w-16 h-16 text-lg',
};

const getInitials = (name: string): string => {
  const parts = name.trim().split(' ');
  if (parts.length === 1) {
    return parts[0].substring(0, 2).toUpperCase();
  }
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
};

const Avatar = ({ src, alt, name, size = 'md', className }: AvatarProps) => {
  if (src) {
    return (
      <img
        src={src}
        alt={alt || name || 'Avatar'}
        className={cn('rounded-full object-cover', sizeClasses[size], className)}
      />
    );
  }

  if (name) {
    return (
      <div
        className={cn(
          'rounded-full bg-blue-600 text-white flex items-center justify-center font-medium',
          sizeClasses[size],
          className
        )}
        aria-label={name}
      >
        {getInitials(name)}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'rounded-full bg-gray-200 text-gray-600 flex items-center justify-center',
        sizeClasses[size],
        className
      )}
      aria-label="Avatar"
    >
      <User className="w-1/2 h-1/2" />
    </div>
  );
};

export { Avatar };

