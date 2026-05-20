'use client';

import React, { forwardRef } from 'react';
import { clsx } from 'clsx';

interface RadioProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  description?: string;
  error?: string;
}

export const Radio = forwardRef<HTMLInputElement, RadioProps>(
  ({ label, description, error, className, ...props }, ref) => {
    return (
      <div className="space-y-1">
        <label className="flex items-start space-x-3 cursor-pointer group">
          <div className="relative flex items-center justify-center flex-shrink-0 mt-0.5">
            <input
              ref={ref}
              type="radio"
              className="sr-only"
              {...props}
            />
            <div
              className={clsx(
                'h-5 w-5 border-2 rounded-full transition-all',
                props.checked
                  ? 'border-primary-600 dark:border-primary-500'
                  : 'border-gray-300 dark:border-gray-700',
                error && 'border-red-500 dark:border-red-500',
                'group-hover:border-primary-500 dark:group-hover:border-primary-400'
              )}
            >
              {props.checked && (
                <div className="h-2.5 w-2.5 bg-primary-600 dark:bg-primary-500 rounded-full absolute inset-0 m-auto" />
              )}
            </div>
          </div>
          <div className="flex-1">
            {label && (
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {label}
                {props.required && <span className="text-red-500 ml-1">*</span>}
              </span>
            )}
            {description && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                {description}
              </p>
            )}
          </div>
        </label>
        {error && (
          <p className="text-sm text-red-600 dark:text-red-400 ml-8">
            {error}
          </p>
        )}
      </div>
    );
  }
);

Radio.displayName = 'Radio';


