/**
 * Responsive container component.
 * Provides consistent responsive layout patterns.
 */

import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface ResponsiveContainerProps {
  children: ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
  padding?: boolean;
}

/**
 * Responsive container component.
 * Provides consistent responsive layout with mobile-first approach.
 *
 * @param props - Component props
 * @returns Responsive container component
 */
export function ResponsiveContainer({
  children,
  className,
  maxWidth = 'xl',
  padding = true,
}: ResponsiveContainerProps) {
  const maxWidthClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-7xl',
    '2xl': 'max-w-7xl',
    full: 'max-w-full',
  };

  return (
    <div
      className={cn(
        'w-full mx-auto',
        maxWidthClasses[maxWidth],
        padding && 'px-4 sm:px-6 lg:px-8',
        className
      )}
    >
      {children}
    </div>
  );
}

interface ResponsiveGridProps {
  children: ReactNode;
  className?: string;
  cols?: {
    default?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
  };
  gap?: 'sm' | 'md' | 'lg';
}

/**
 * Responsive grid component.
 * Provides responsive grid layout with mobile-first breakpoints.
 *
 * @param props - Component props
 * @returns Responsive grid component
 */
export function ResponsiveGrid({
  children,
  className,
  cols = { default: 1, sm: 2, md: 3, lg: 4 },
  gap = 'md',
}: ResponsiveGridProps) {
  const gapClasses = {
    sm: 'gap-2',
    md: 'gap-4',
    lg: 'gap-6',
  };

  const gridColsClasses = {
    default: `grid-cols-${cols.default || 1}`,
    sm: cols.sm ? `sm:grid-cols-${cols.sm}` : '',
    md: cols.md ? `md:grid-cols-${cols.md}` : '',
    lg: cols.lg ? `lg:grid-cols-${cols.lg}` : '',
    xl: cols.xl ? `xl:grid-cols-${cols.xl}` : '',
  };

  return (
    <div
      className={cn(
        'grid',
        gridColsClasses.default,
        gridColsClasses.sm,
        gridColsClasses.md,
        gridColsClasses.lg,
        gridColsClasses.xl,
        gapClasses[gap],
        className
      )}
    >
      {children}
    </div>
  );
}

