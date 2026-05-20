'use client';

import { forwardRef } from 'react';
import { Textarea } from './textarea';
import { FormField } from './form-field';
import { cn } from '@/lib/utils/cn';
import type { TextareaProps } from './textarea';

interface FormTextareaProps extends TextareaProps {
  label: string;
  error?: string;
  required?: boolean;
}

export const FormTextarea = forwardRef<HTMLTextAreaElement, FormTextareaProps>(
  ({ label, error, required, className, id, ...props }, ref) => {
    return (
      <FormField
        label={label}
        htmlFor={id || ''}
        error={error}
        required={required}
      >
        <Textarea
          ref={ref}
          id={id}
          className={cn(error ? 'border-red-500' : '', className)}
          {...props}
        />
      </FormField>
    );
  }
);

FormTextarea.displayName = 'FormTextarea';

