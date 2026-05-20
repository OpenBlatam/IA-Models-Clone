/**
 * Button Component
 * Optimized button component with variants, sizes, and loading states
 */

import { ButtonHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';
import { Spinner } from './Spinner';
import { BUTTON_VARIANTS, BUTTON_SIZES } from '@/lib/constants/ui';

type ButtonVariant = (typeof BUTTON_VARIANTS)[keyof typeof BUTTON_VARIANTS];
type ButtonSize = (typeof BUTTON_SIZES)[keyof typeof BUTTON_SIZES];

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  fullWidth?: boolean;
}

/**
 * Button component with multiple variants and sizes
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = BUTTON_VARIANTS.PRIMARY,
      size = BUTTON_SIZES.MD,
      loading = false,
      disabled,
      children,
      fullWidth = false,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none disabled:cursor-not-allowed';

    const variants: Record<ButtonVariant, string> = {
      [BUTTON_VARIANTS.PRIMARY]:
        'bg-primary-600 text-white hover:bg-primary-700 active:bg-primary-800 focus:ring-primary-500 dark:bg-primary-500 dark:hover:bg-primary-600 dark:active:bg-primary-700',
      [BUTTON_VARIANTS.SECONDARY]:
        'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 active:bg-gray-400 dark:active:bg-gray-500 focus:ring-gray-500',
      [BUTTON_VARIANTS.DANGER]:
        'bg-red-600 text-white hover:bg-red-700 active:bg-red-800 focus:ring-red-500 dark:bg-red-500 dark:hover:bg-red-600 dark:active:bg-red-700',
      [BUTTON_VARIANTS.GHOST]:
        'bg-transparent text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 active:bg-gray-200 dark:active:bg-gray-700 focus:ring-gray-500',
      [BUTTON_VARIANTS.OUTLINE]:
        'border-2 border-primary-600 text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 active:bg-primary-100 dark:active:bg-primary-900/30 focus:ring-primary-500 dark:border-primary-500 dark:text-primary-500',
      [BUTTON_VARIANTS.LINK]:
        'text-primary-600 hover:text-primary-700 underline-offset-4 hover:underline focus:ring-primary-500 dark:text-primary-400 dark:hover:text-primary-300',
    };

    const sizes: Record<ButtonSize, string> = {
      [BUTTON_SIZES.SM]: 'px-3 py-1.5 text-sm gap-1.5',
      [BUTTON_SIZES.MD]: 'px-4 py-2 text-sm gap-2',
      [BUTTON_SIZES.LG]: 'px-6 py-3 text-base gap-2',
      [BUTTON_SIZES.ICON]: 'p-2',
    };

    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        className={cn(
          baseStyles,
          variants[variant],
          sizes[size],
          fullWidth && 'w-full',
          className
        )}
        disabled={isDisabled}
        aria-busy={loading}
        aria-disabled={isDisabled}
        {...props}
      >
        {loading && (
          <Spinner
            size={size === BUTTON_SIZES.LG ? 'md' : 'sm'}
            variant={variant === BUTTON_VARIANTS.GHOST || variant === BUTTON_VARIANTS.OUTLINE ? 'primary' : 'white'}
            className="shrink-0"
            aria-hidden="true"
          />
        )}
        {children && <span className={loading ? 'opacity-70' : ''}>{children}</span>}
      </button>
    );
  }
);

Button.displayName = 'Button';

