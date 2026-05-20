/**
 * Button component.
 * Reusable button component with variants and sizes.
 */

import { type ButtonHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

/**
 * Button variants.
 */
export type ButtonVariant =
  | 'primary'
  | 'secondary'
  | 'outline'
  | 'ghost'
  | 'danger';

/**
 * Button sizes.
 */
export type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
  fullWidth?: boolean;
}

/**
 * Button component.
 * Provides consistent button styling with variants and loading states.
 *
 * @param props - Component props
 * @returns Button component
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      isLoading = false,
      fullWidth = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const variantClasses = {
      primary:
        'bg-purple-600 hover:bg-purple-700 text-white focus:ring-purple-400',
      secondary:
        'bg-white/10 hover:bg-white/20 text-white focus:ring-purple-400',
      outline:
        'border border-white/20 hover:bg-white/10 text-white focus:ring-purple-400',
      ghost: 'hover:bg-white/10 text-white focus:ring-purple-400',
      danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-400',
    };

    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg',
    };

    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-2',
          'rounded-lg font-medium transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-purple-900',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          variantClasses[variant],
          sizeClasses[size],
          fullWidth && 'w-full',
          className
        )}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading && (
          <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
        )}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

