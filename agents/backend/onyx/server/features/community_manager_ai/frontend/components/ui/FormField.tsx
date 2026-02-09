'use client';

import { ReactNode } from 'react';
import { Label } from './Label';
import { cn } from '@/lib/utils';

interface FormFieldProps {
  label?: string;
  required?: boolean;
  error?: string;
  hint?: string;
  children: ReactNode;
  className?: string;
  htmlFor?: string;
}

export const FormField = ({
  label,
  required,
  error,
  hint,
  children,
  className,
  htmlFor,
}: FormFieldProps) => {
  return (
    <div className={cn('space-y-2', className)}>
      {label && (
        <Label htmlFor={htmlFor} required={required} error={!!error}>
          {label}
        </Label>
      )}
      {children}
      {error && (
        <p className="text-sm text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
      {hint && !error && (
        <p className="text-sm text-gray-500 dark:text-gray-400">{hint}</p>
      )}
    </div>
  );
};



