'use client';

import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

interface CodeProps extends React.HTMLAttributes<HTMLElement> {
  inline?: boolean;
}

export const Code = forwardRef<HTMLElement, CodeProps>(
  ({ inline = false, className, children, ...props }, ref) => {
    if (inline) {
      return (
        <code
          ref={ref}
          className={cn(
            'rounded bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 text-sm font-mono',
            'text-gray-900 dark:text-gray-100',
            className
          )}
          {...props}
        >
          {children}
        </code>
      );
    }

    return (
      <code
        ref={ref}
        className={cn(
          'block rounded-lg bg-gray-900 dark:bg-gray-950 p-4 text-sm font-mono',
          'text-gray-100 overflow-x-auto',
          className
        )}
        {...props}
      >
        {children}
      </code>
    );
  }
);

Code.displayName = 'Code';



