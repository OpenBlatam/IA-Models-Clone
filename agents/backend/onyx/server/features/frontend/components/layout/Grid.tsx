'use client';

import { HTMLAttributes } from 'react';

interface GridProps extends HTMLAttributes<HTMLDivElement> {
  cols?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
  gap?: 'none' | 'sm' | 'md' | 'lg';
  responsive?: {
    sm?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
    md?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
    lg?: 1 | 2 | 3 | 4 | 5 | 6 | 12;
  };
}

const gapClasses = {
  none: 'gap-0',
  sm: 'gap-2',
  md: 'gap-4',
  lg: 'gap-6',
};

const getColClass = (cols: number) => {
  const colMap: Record<number, string> = {
    1: 'grid-cols-1',
    2: 'grid-cols-2',
    3: 'grid-cols-3',
    4: 'grid-cols-4',
    5: 'grid-cols-5',
    6: 'grid-cols-6',
    12: 'grid-cols-12',
  };
  return colMap[cols] || 'grid-cols-1';
};

export function Grid({
  children,
  cols = 1,
  gap = 'md',
  responsive,
  className = '',
  ...props
}: GridProps) {
  const baseClass = getColClass(cols);
  const responsiveClasses = responsive
    ? `
      ${responsive.sm ? `sm:${getColClass(responsive.sm)}` : ''}
      ${responsive.md ? `md:${getColClass(responsive.md)}` : ''}
      ${responsive.lg ? `lg:${getColClass(responsive.lg)}` : ''}
    `.trim()
    : '';

  return (
    <div
      className={`
        grid
        ${baseClass}
        ${responsiveClasses}
        ${gapClasses[gap]}
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}

