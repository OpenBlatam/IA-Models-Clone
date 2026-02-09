'use client';

import React from 'react';
import { clsx } from 'clsx';

interface HelperTextProps {
  children: React.ReactNode;
  error?: boolean;
  className?: string;
}

export const HelperText: React.FC<HelperTextProps> = ({
  children,
  error = false,
  className,
}) => {
  return (
    <p
      className={clsx(
        'text-sm mt-1',
        error
          ? 'text-red-600 dark:text-red-400'
          : 'text-gray-500 dark:text-gray-400',
        className
      )}
    >
      {children}
    </p>
  );
};


