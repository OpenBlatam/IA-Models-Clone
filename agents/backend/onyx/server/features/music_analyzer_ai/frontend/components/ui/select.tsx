/**
 * Select component.
 * Reusable select component with validation states.
 */

import { type SelectHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  error?: boolean;
  fullWidth?: boolean;
}

/**
 * Select component.
 * Provides consistent select styling with validation states.
 *
 * @param props - Component props
 * @returns Select component
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, error = false, fullWidth = false, children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={cn(
          'px-4 py-2',
          'bg-white/10 border rounded-lg',
          'text-white',
          'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-slate-900',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'transition-colors',
          'appearance-none bg-no-repeat bg-right',
          'pr-10',
          error
            ? 'border-red-500 focus:ring-red-400'
            : 'border-white/20 focus:border-purple-400',
          fullWidth && 'w-full',
          className
        )}
        style={{
          backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%23ffffff' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`,
          backgroundPosition: 'right 0.5rem center',
          backgroundSize: '1.5em 1.5em',
        }}
        {...props}
      >
        {children}
      </select>
    );
  }
);
Select.displayName = 'Select';

