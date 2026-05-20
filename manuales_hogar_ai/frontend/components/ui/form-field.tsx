'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils/cn';

interface FormFieldProps {
  label: string;
  htmlFor: string;
  error?: string;
  required?: boolean;
  children: ReactNode;
  className?: string;
}

export const FormField = ({
  label,
  htmlFor,
  error,
  required = false,
  children,
  className,
}: FormFieldProps): JSX.Element => {
  return (
    <div className={className}>
      <label htmlFor={htmlFor} className="block text-sm font-medium mb-2">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {children}
      {error && (
        <p className="text-sm text-red-500 mt-1">{error}</p>
      )}
    </div>
  );
};

interface FormFieldWrapperProps {
  children: ReactNode;
  className?: string;
}

export const FormFieldWrapper = ({
  children,
  className,
}: FormFieldWrapperProps): JSX.Element => {
  return <div className={cn('grid grid-cols-2 gap-4', className)}>{children}</div>;
};

