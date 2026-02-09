'use client';

import { memo, useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import Image from 'next/image';

interface AvatarProps {
  src?: string;
  alt?: string;
  fallback?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  variant?: 'circle' | 'square' | 'rounded';
  status?: 'online' | 'offline' | 'away' | 'busy';
}

const Avatar = memo(
  ({
    src,
    alt,
    fallback,
    size = 'md',
    className,
    variant = 'circle',
    status,
  }: AvatarProps): JSX.Element => {
    const [hasError, setHasError] = useState(false);

    const sizeClasses = {
      sm: 'w-8 h-8 text-sm',
      md: 'w-10 h-10 text-base',
      lg: 'w-12 h-12 text-lg',
      xl: 'w-16 h-16 text-xl',
    };

    const variantClasses = {
      circle: 'rounded-full',
      square: 'rounded-none',
      rounded: 'rounded-lg',
    };

    const statusColors = {
      online: 'bg-green-500',
      offline: 'bg-gray-400',
      away: 'bg-yellow-500',
      busy: 'bg-red-500',
    };

    const handleError = useCallback((): void => {
      setHasError(true);
    }, []);

    const getInitials = useCallback((name: string): string => {
      const words = name.trim().split(/\s+/);
      if (words.length >= 2) {
        return (words[0][0] + words[words.length - 1][0]).toUpperCase();
      }
      return name.substring(0, 2).toUpperCase();
    }, []);

    const displayFallback = fallback ? getInitials(fallback) : '?';

    return (
      <div
        className={cn('relative inline-block', className)}
        role="img"
        aria-label={alt || 'Avatar'}
      >
        <div
          className={cn(
            'bg-gray-200 flex items-center justify-center font-semibold text-gray-600 overflow-hidden',
            sizeClasses[size],
            variantClasses[variant],
            className
          )}
        >
          {src && !hasError ? (
            <Image
              src={src}
              alt={alt || 'Avatar'}
              fill
              className="object-cover"
              onError={handleError}
              unoptimized
            />
          ) : (
            <span className="select-none">{displayFallback}</span>
          )}
        </div>
        {status && (
          <span
            className={cn(
              'absolute bottom-0 right-0 rounded-full border-2 border-white',
              statusColors[status],
              size === 'sm' ? 'w-2.5 h-2.5' : size === 'md' ? 'w-3 h-3' : 'w-3.5 h-3.5'
            )}
            aria-label={`Status: ${status}`}
          />
        )}
      </div>
    );
  }
);

Avatar.displayName = 'Avatar';

export default Avatar;

