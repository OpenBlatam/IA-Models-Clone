'use client';

import React from 'react';
import { clsx } from 'clsx';

interface SeparatorProps {
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export const Separator: React.FC<SeparatorProps> = ({
  orientation = 'horizontal',
  className,
}) => {
  return (
    <div
      className={clsx(
        'bg-gray-200 dark:bg-gray-700',
        orientation === 'horizontal' ? 'h-px w-full' : 'w-px h-full',
        className
      )}
      role="separator"
      aria-orientation={orientation}
    />
  );
};
