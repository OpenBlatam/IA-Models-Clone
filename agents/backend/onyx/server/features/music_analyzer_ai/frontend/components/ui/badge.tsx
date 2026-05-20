/**
 * Badge component.
 * Reusable badge component with variants and sizes.
 */

import { type HTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

/**
 * Badge variants.
 */
export type BadgeVariant =
  | 'default'
  | 'primary'
  | 'secondary'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info';

/**
 * Badge sizes.
 */
export type BadgeSize = 'sm' | 'md' | 'lg';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  rounded?: boolean;
}

/**
 * Badge component.
 * Provides consistent badge styling with variants.
 *
 * @param props - Component props
 * @returns Badge component
 */
export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  (
    { className, variant = 'default', size = 'md', rounded = false, ...props },
    ref
  ) => {
    const variantClasses = {
      default: 'bg-gray-600 text-white',
      primary: 'bg-purple-600 text-white',
      secondary: 'bg-white/10 text-white',
      success: 'bg-green-600 text-white',
      warning: 'bg-yellow-600 text-white',
      danger: 'bg-red-600 text-white',
      info: 'bg-blue-600 text-white',
    };

    const sizeClasses = {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-2.5 py-1 text-sm',
      lg: 'px-3 py-1.5 text-base',
    };

    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center font-medium',
          variantClasses[variant],
          sizeClasses[size],
          rounded ? 'rounded-full' : 'rounded-md',
          className
        )}
        {...props}
      />
    );
  }
);

Badge.displayName = 'Badge';

