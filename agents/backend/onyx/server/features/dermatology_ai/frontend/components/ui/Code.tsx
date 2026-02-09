'use client';

import React from 'react';
import { clsx } from 'clsx';

interface CodeProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'inline' | 'block';
}

export const Code: React.FC<CodeProps> = ({
  children,
  className,
  variant = 'inline',
}) => {
  if (variant === 'block') {
    return (
      <pre className={clsx('p-4 bg-gray-100 dark:bg-gray-800 rounded-lg overflow-x-auto', className)}>
        <code className="text-sm font-mono text-gray-900 dark:text-gray-100">
          {children}
        </code>
      </pre>
    );
  }

  return (
    <code
      className={clsx(
        'px-1.5 py-0.5 text-sm font-mono bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded',
        className
      )}
    >
      {children}
    </code>
  );
};
