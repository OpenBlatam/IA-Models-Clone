'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  spacing?: 'none' | 'sm' | 'md' | 'lg';
  variant?: 'solid' | 'dashed' | 'dotted';
  className?: string;
}

const spacingClasses = {
  none: '',
  sm: 'my-2',
  md: 'my-4',
  lg: 'my-6',
};

const variantClasses = {
  solid: 'border-solid',
  dashed: 'border-dashed',
  dotted: 'border-dotted',
};

export const Divider: React.FC<DividerProps> = memo(({
  orientation = 'horizontal',
  spacing = 'md',
  variant = 'solid',
  className,
}) => {
  if (orientation === 'vertical') {
    return (
      <div
        className={clsx(
          'border-l border-gray-200 dark:border-gray-700',
          variantClasses[variant],
          spacing === 'sm' && 'mx-2',
          spacing === 'md' && 'mx-4',
          spacing === 'lg' && 'mx-6',
          className
        )}
        aria-hidden="true"
      />
    );
  }

  return (
    <hr
      className={clsx(
        'border-0 border-t border-gray-200 dark:border-gray-700',
        variantClasses[variant],
        spacingClasses[spacing],
        className
      )}
      aria-hidden="true"
    />
  );
});

Divider.displayName = 'Divider';
