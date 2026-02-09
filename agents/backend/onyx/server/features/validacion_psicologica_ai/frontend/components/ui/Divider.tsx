/**
 * Divider component
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface DividerProps extends React.HTMLAttributes<HTMLHRElement> {
  orientation?: 'horizontal' | 'vertical';
  label?: string;
}

const Divider = React.forwardRef<HTMLHRElement, DividerProps>(
  ({ className, orientation = 'horizontal', label, ...props }, ref) => {
    if (orientation === 'vertical') {
      return (
        <div
          ref={ref as React.RefObject<HTMLDivElement>}
          className={cn('w-px bg-border self-stretch', className)}
          role="separator"
          aria-orientation="vertical"
          {...(props as React.HTMLAttributes<HTMLDivElement>)}
        />
      );
    }

    if (label) {
      return (
        <div className={cn('flex items-center gap-4 my-4', className)}>
          <hr
            ref={ref}
            className="flex-1 border-t border-border"
            role="separator"
            aria-orientation="horizontal"
            {...props}
          />
          <span className="text-sm text-muted-foreground whitespace-nowrap">{label}</span>
          <hr
            className="flex-1 border-t border-border"
            role="separator"
            aria-orientation="horizontal"
          />
        </div>
      );
    }

    return (
      <hr
        ref={ref}
        className={cn('border-t border-border my-4', className)}
        role="separator"
        aria-orientation="horizontal"
        {...props}
      />
    );
  }
);

Divider.displayName = 'Divider';

export { Divider };
