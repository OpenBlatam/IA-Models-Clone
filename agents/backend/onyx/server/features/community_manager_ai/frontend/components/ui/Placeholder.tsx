'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface PlaceholderProps {
  children: ReactNode;
  className?: string;
  variant?: 'default' | 'dashed' | 'dotted';
}

const variantClasses = {
  default: 'border-gray-300 dark:border-gray-700',
  dashed: 'border-dashed border-gray-300 dark:border-gray-700',
  dotted: 'border-dotted border-gray-300 dark:border-gray-700',
};

export const Placeholder = ({
  children,
  className,
  variant = 'default',
}: PlaceholderProps) => {
  return (
    <div
      className={cn(
        'rounded-lg border-2 p-8 text-center',
        variantClasses[variant],
        className
      )}
    >
      {children}
    </div>
  );
};



