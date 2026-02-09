/**
 * Form field component wrapper
 */

import React from 'react';
import { Label } from './Label';
import { cn } from '@/lib/utils/cn';

export interface FormFieldProps {
  label?: string;
  required?: boolean;
  error?: string;
  helperText?: string;
  children: React.ReactNode;
  className?: string;
  id?: string;
}

export const FormField: React.FC<FormFieldProps> = ({
  label,
  required,
  error,
  helperText,
  children,
  className,
  id,
}) => {
  const fieldId = id || `field-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div className={cn('space-y-2', className)}>
      {label && (
        <Label htmlFor={fieldId} required={required} error={!!error}>
          {label}
        </Label>
      )}
      {React.cloneElement(children as React.ReactElement, {
        id: fieldId,
        'aria-describedby': error
          ? `${fieldId}-error`
          : helperText
          ? `${fieldId}-helper`
          : undefined,
        'aria-invalid': error ? 'true' : 'false',
      })}
      {error && (
        <p
          id={`${fieldId}-error`}
          className="text-sm text-destructive"
          role="alert"
        >
          {error}
        </p>
      )}
      {helperText && !error && (
        <p id={`${fieldId}-helper`} className="text-sm text-muted-foreground">
          {helperText}
        </p>
      )}
    </div>
  );
};



