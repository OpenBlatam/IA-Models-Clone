/**
 * Spinner Component
 * Optimized loading spinner with variants
 */

import { cn } from '@/lib/utils';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'white' | 'gray';
  className?: string;
  'aria-label'?: string;
}

/**
 * Loading spinner component
 */
export const Spinner = ({
  size = 'md',
  variant = 'primary',
  className,
  'aria-label': ariaLabel = 'Cargando...',
}: SpinnerProps) => {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
  };

  const variants = {
    primary: 'border-primary-600',
    white: 'border-white',
    gray: 'border-gray-600',
  };

  return (
    <div
      className={cn(
        'animate-spin rounded-full border-2 border-t-transparent',
        sizes[size],
        variants[variant],
        className
      )}
      role="status"
      aria-label={ariaLabel}
    >
      <span className="sr-only">{ariaLabel}</span>
    </div>
  );
};
