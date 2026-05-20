'use client';

/**
 * Form field component with built-in validation.
 * Provides consistent form field styling and error display.
 */

import { forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { AlertCircle } from 'lucide-react';

export interface FormFieldProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  /**
   * Label for the field.
   */
  label?: string;
  /**
   * Error messages to display.
   */
  errors?: string[];
  /**
   * Whether the field has been touched.
   */
  touched?: boolean;
  /**
   * Helper text to display below the field.
   */
  helperText?: string;
  /**
   * Whether the field is required.
   */
  required?: boolean;
}

/**
 * Form field component with validation support.
 */
export const FormField = forwardRef<HTMLInputElement, FormFieldProps>(
  (
    {
      label,
      errors = [],
      touched = false,
      helperText,
      required = false,
      className,
      id,
      ...props
    },
    ref
  ) => {
    const fieldId = id || `field-${label?.toLowerCase().replace(/\s+/g, '-')}`;
    const hasError = touched && errors.length > 0;
    const showError = hasError;
    const showHelper = !showError && helperText;

    return (
      <div className="space-y-2">
        {label && (
          <label
            htmlFor={fieldId}
            className="block text-sm font-medium text-gray-300"
          >
            {label}
            {required && <span className="text-red-400 ml-1">*</span>}
          </label>
        )}
        <div className="relative">
          <input
            ref={ref}
            id={fieldId}
            className={cn(
              'w-full px-4 py-2 bg-white/10 border rounded-lg',
              'text-white placeholder-gray-400',
              'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-transparent',
              'transition-colors',
              hasError
                ? 'border-red-500 focus:ring-red-500'
                : 'border-white/20',
              className
            )}
            aria-invalid={hasError}
            aria-describedby={
              showError
                ? `${fieldId}-error`
                : showHelper
                ? `${fieldId}-helper`
                : undefined
            }
            {...props}
          />
          {hasError && (
            <AlertCircle className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-red-400" />
          )}
        </div>
        {showError && (
          <div
            id={`${fieldId}-error`}
            className="text-sm text-red-400 space-y-1"
            role="alert"
          >
            {errors.map((error, index) => (
              <p key={index}>{error}</p>
            ))}
          </div>
        )}
        {showHelper && (
          <p id={`${fieldId}-helper`} className="text-sm text-gray-400">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

FormField.displayName = 'FormField';

