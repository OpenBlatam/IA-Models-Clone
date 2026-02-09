'use client';

import React from 'react';
import { clsx } from 'clsx';

type GridCols = 1 | 2 | 3 | 4 | 5 | 6 | 12;
type GridBreakpoint = 'sm' | 'md' | 'lg' | 'xl' | '2xl';

interface GridProps {
  children: React.ReactNode;
  cols?: GridCols | { base?: GridCols; sm?: GridCols; md?: GridCols; lg?: GridCols; xl?: GridCols; '2xl'?: GridCols };
  gap?: 2 | 4 | 6 | 8;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '4xl' | '5xl' | '6xl' | '7xl' | 'full' | 'none';
}

const maxWidthClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '4xl': 'max-w-4xl',
  '5xl': 'max-w-5xl',
  '6xl': 'max-w-6xl',
  '7xl': 'max-w-7xl',
  full: 'max-w-full',
  none: '',
};

const gridColsMap: Record<GridCols, string> = {
  1: 'grid-cols-1',
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-4',
  5: 'grid-cols-5',
  6: 'grid-cols-6',
  12: 'grid-cols-12',
};

const getGridColsClass = (cols: GridCols | { base?: GridCols; sm?: GridCols; md?: GridCols; lg?: GridCols; xl?: GridCols; '2xl'?: GridCols }): string => {
  if (typeof cols === 'number') {
    return gridColsMap[cols];
  }
  
  const classes: string[] = [];
  if (cols.base) classes.push(gridColsMap[cols.base]);
  if (cols.sm) classes.push(`sm:${gridColsMap[cols.sm]}`);
  if (cols.md) classes.push(`md:${gridColsMap[cols.md]}`);
  if (cols.lg) classes.push(`lg:${gridColsMap[cols.lg]}`);
  if (cols.xl) classes.push(`xl:${gridColsMap[cols.xl]}`);
  if (cols['2xl']) classes.push(`2xl:${gridColsMap[cols['2xl']]}`);
  
  return classes.join(' ');
};

export const Grid: React.FC<GridProps> = ({
  children,
  cols = 1,
  gap = 6,
  className,
  maxWidth = 'none',
}) => {
  const gridColsClass = getGridColsClass(cols);
  
  const gapMap: Record<typeof gap, string> = {
    2: 'gap-2',
    4: 'gap-4',
    6: 'gap-6',
    8: 'gap-8',
  };

  return (
    <div
      className={clsx(
        'grid',
        gridColsClass,
        gapMap[gap],
        maxWidth !== 'none' && maxWidthClasses[maxWidth],
        maxWidth !== 'none' && 'mx-auto',
        className
      )}
    >
      {children}
    </div>
  );
};

