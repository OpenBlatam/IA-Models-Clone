/**
 * Avatar component
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {
  src?: string;
  alt?: string;
  fallback?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  shape?: 'circle' | 'square';
}

const sizeClasses = {
  sm: 'h-8 w-8 text-xs',
  md: 'h-10 w-10 text-sm',
  lg: 'h-12 w-12 text-base',
  xl: 'h-16 w-16 text-lg',
};

export const Avatar: React.FC<AvatarProps> = ({
  src,
  alt,
  fallback,
  size = 'md',
  shape = 'circle',
  className,
  ...props
}) => {
  const [imageError, setImageError] = React.useState(false);
  const showImage = src && !imageError;

  const getInitials = (text?: string): string => {
    if (!text) {
      return '?';
    }
    const words = text.trim().split(/\s+/);
    if (words.length >= 2) {
      return (words[0][0] + words[1][0]).toUpperCase();
    }
    return text[0].toUpperCase();
  };

  return (
    <div
      className={cn(
        'relative flex items-center justify-center bg-muted text-muted-foreground font-medium overflow-hidden',
        sizeClasses[size],
        shape === 'circle' ? 'rounded-full' : 'rounded-md',
        className
      )}
      {...props}
    >
      {showImage ? (
        <img
          src={src}
          alt={alt}
          className={cn('w-full h-full object-cover', shape === 'circle' ? 'rounded-full' : 'rounded-md')}
          onError={() => setImageError(true)}
        />
      ) : (
        <span aria-label={alt || 'Avatar'}>{fallback || getInitials(alt)}</span>
      )}
    </div>
  );
};
