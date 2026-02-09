/**
 * Checkbox component.
 * Reusable checkbox component with validation states.
 */

import { type InputHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface CheckboxProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  error?: boolean;
}

/**
 * Checkbox component.
 * Provides consistent checkbox styling with label.
 *
 * @param props - Component props
 * @returns Checkbox component
 */
export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, error = false, ...props }, ref) => {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          ref={ref}
          type="checkbox"
          className={cn(
            'w-4 h-4',
            'rounded border-white/20',
            'bg-white/10',
            'text-purple-600',
            'focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-slate-900',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'transition-colors',
            error && 'border-red-500',
            className
          )}
          {...props}
        />
        {label && (
          <span className={cn('text-sm', error && 'text-red-400')}>
            {label}
          </span>
        )}
      </label>
    );
  }
);
Checkbox.displayName = 'Checkbox';

