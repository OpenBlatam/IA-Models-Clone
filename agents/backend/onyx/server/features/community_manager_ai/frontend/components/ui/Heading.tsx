'use client';

import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

interface HeadingProps extends React.HTMLAttributes<HTMLHeadingElement> {
  as?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
  level?: 1 | 2 | 3 | 4 | 5 | 6;
}

const sizeClasses = {
  1: 'text-4xl font-bold',
  2: 'text-3xl font-bold',
  3: 'text-2xl font-semibold',
  4: 'text-xl font-semibold',
  5: 'text-lg font-medium',
  6: 'text-base font-medium',
};

export const Heading = forwardRef<HTMLHeadingElement, HeadingProps>(
  ({ as, level = 1, className, children, ...props }, ref) => {
    const Component = as || (`h${level}` as 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6');
    const sizeClass = sizeClasses[level];

    if (Component === 'h1') {
      return <h1 ref={ref} className={cn(sizeClass, 'text-gray-900 dark:text-gray-100', className)} {...props}>{children}</h1>;
    }
    if (Component === 'h2') {
      return <h2 ref={ref} className={cn(sizeClass, 'text-gray-900 dark:text-gray-100', className)} {...props}>{children}</h2>;
    }
    if (Component === 'h3') {
      return <h3 ref={ref} className={cn(sizeClass, 'text-gray-900 dark:text-gray-100', className)} {...props}>{children}</h3>;
    }
    if (Component === 'h4') {
      return <h4 ref={ref} className={cn(sizeClass, 'text-gray-900 dark:text-gray-100', className)} {...props}>{children}</h4>;
    }
    if (Component === 'h5') {
      return <h5 ref={ref} className={cn(sizeClass, 'text-gray-900 dark:text-gray-100', className)} {...props}>{children}</h5>;
    }
    return <h6 ref={ref} className={cn(sizeClass, 'text-gray-900 dark:text-gray-100', className)} {...props}>{children}</h6>;
  }
);

Heading.displayName = 'Heading';

