/**
 * Switch component.
 * Reusable switch/toggle component.
 */

import { type InputHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface SwitchProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  error?: boolean;
}

/**
 * Switch component.
 * Provides consistent switch styling with label.
 *
 * @param props - Component props
 * @returns Switch component
 */
export const Switch = forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, label, error = false, ...props }, ref) => {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          ref={ref}
          type="checkbox"
          role="switch"
          className={cn(
            'sr-only',
            className
          )}
          {...props}
        />
        <div
          className={cn(
            'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
            'focus-within:ring-2 focus-within:ring-purple-400 focus-within:ring-offset-2',
            props.checked
              ? 'bg-purple-600'
              : 'bg-gray-600',
            props.disabled && 'opacity-50 cursor-not-allowed',
            error && 'ring-2 ring-red-500'
          )}
        >
          <span
            className={cn(
              'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
              props.checked ? 'translate-x-6' : 'translate-x-1'
            )}
          />
        </div>
        {label && (
          <span className={cn('text-sm', error && 'text-red-400')}>
            {label}
          </span>
        )}
      </label>
    );
  }
);
Switch.displayName = 'Switch';

