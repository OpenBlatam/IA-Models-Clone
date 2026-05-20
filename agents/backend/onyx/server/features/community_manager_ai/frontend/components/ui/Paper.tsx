'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface PaperProps {
  children: ReactNode;
  elevation?: 0 | 1 | 2 | 3 | 4;
  variant?: 'elevation' | 'outlined';
  className?: string;
}

const elevationClasses = {
  0: '',
  1: 'shadow-sm',
  2: 'shadow-md',
  3: 'shadow-lg',
  4: 'shadow-xl',
};

export const Paper = ({
  children,
  elevation = 1,
  variant = 'elevation',
  className,
}: PaperProps) => {
  return (
    <div
      className={cn(
        'rounded-lg bg-white dark:bg-gray-800',
        variant === 'elevation' && elevationClasses[elevation],
        variant === 'outlined' && 'border border-gray-200 dark:border-gray-700',
        className
      )}
    >
      {children}
    </div>
  );
};



