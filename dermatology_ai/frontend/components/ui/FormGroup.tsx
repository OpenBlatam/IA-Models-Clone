'use client';

import React from 'react';
import { clsx } from 'clsx';

interface FormGroupProps {
  children: React.ReactNode;
  className?: string;
  error?: boolean;
}

export const FormGroup: React.FC<FormGroupProps> = ({
  children,
  className,
  error,
}) => {
  return (
    <div
      className={clsx(
        'space-y-2',
        error && 'text-red-600 dark:text-red-400',
        className
      )}
    >
      {children}
    </div>
  );
};


