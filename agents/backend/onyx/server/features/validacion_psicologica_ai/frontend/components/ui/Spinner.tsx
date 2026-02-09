/**
 * Spinner component with variants
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';
import { cva, type VariantProps } from 'class-variance-authority';

const spinnerVariants = cva('animate-spin rounded-full border-2 border-t-transparent', {
  variants: {
    size: {
      sm: 'h-4 w-4 border-2',
      md: 'h-8 w-8 border-2',
      lg: 'h-12 w-12 border-4',
    },
    variant: {
      primary: 'border-primary',
      secondary: 'border-secondary-foreground',
      muted: 'border-muted-foreground',
    },
  },
  defaultVariants: {
    size: 'md',
    variant: 'primary',
  },
});

export interface SpinnerProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof spinnerVariants> {
  label?: string;
}

const Spinner = React.forwardRef<HTMLDivElement, SpinnerProps>(
  ({ className, size, variant, label, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn('flex flex-col items-center justify-center gap-2', className)}
        role="status"
        aria-live="polite"
        aria-label={label || 'Cargando'}
        {...props}
      >
        <div className={cn(spinnerVariants({ size, variant }))} aria-hidden="true" />
        {label && <span className="text-sm text-muted-foreground">{label}</span>}
      </div>
    );
  }
);

Spinner.displayName = 'Spinner';

export { Spinner };
