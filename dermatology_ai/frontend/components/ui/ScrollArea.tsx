'use client';

import React from 'react';
import { clsx } from 'clsx';

interface ScrollAreaProps {
  children: React.ReactNode;
  className?: string;
  orientation?: 'vertical' | 'horizontal' | 'both';
}

export const ScrollArea: React.FC<ScrollAreaProps> = ({
  children,
  className,
  orientation = 'vertical',
}) => {
  const orientationClasses = {
    vertical: 'overflow-y-auto',
    horizontal: 'overflow-x-auto',
    both: 'overflow-auto',
  };

  return (
    <div
      className={clsx(
        'scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 scrollbar-track-gray-100 dark:scrollbar-track-gray-800',
        orientationClasses[orientation],
        className
      )}
    >
      {children}
    </div>
  );
};
