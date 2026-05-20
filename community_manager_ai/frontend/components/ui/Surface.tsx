'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface SurfaceProps {
  children: ReactNode;
  variant?: 'flat' | 'elevated' | 'outlined';
  className?: string;
}

const variantClasses = {
  flat: 'bg-white dark:bg-gray-800',
  elevated: 'bg-white dark:bg-gray-800 shadow-lg',
  outlined: 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
};

export const Surface = ({
  children,
  variant = 'flat',
  className,
}: SurfaceProps) => {
  return (
    <div
      className={cn(
        'rounded-lg',
        variantClasses[variant],
        className
      )}
    >
      {children}
    </div>
  );
};



