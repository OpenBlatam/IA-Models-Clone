'use client';

import React from 'react';
import { clsx } from 'clsx';

interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {
  required?: boolean;
  error?: boolean;
}

export const Label: React.FC<LabelProps> = ({
  children,
  required,
  error,
  className,
  ...props
}) => {
  return (
    <label
      className={clsx(
        'block text-sm font-medium',
        error
          ? 'text-red-600 dark:text-red-400'
          : 'text-gray-700 dark:text-gray-300',
        className
      )}
      {...props}
    >
      {children}
      {required && <span className="text-red-500 ml-1">*</span>}
    </label>
  );
};


