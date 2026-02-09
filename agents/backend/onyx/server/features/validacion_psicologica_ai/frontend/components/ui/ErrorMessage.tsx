/**
 * Error message component with accessibility
 */

import React from 'react';
import { cn } from '@/lib/utils/cn';
import { AlertCircle } from 'lucide-react';

export interface ErrorMessageProps extends React.HTMLAttributes<HTMLDivElement> {
  message: string;
  title?: string;
}

const ErrorMessage = React.forwardRef<HTMLDivElement, ErrorMessageProps>(
  ({ className, message, title, ...props }, ref) => {
    if (!message) {
      return null;
    }

    return (
      <div
        ref={ref}
        className={cn(
          'flex items-start gap-2 p-4 rounded-md bg-destructive/10 border border-destructive/20',
          className
        )}
        role="alert"
        aria-live="assertive"
        {...props}
      >
        <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" aria-hidden="true" />
        <div className="flex-1">
          {title && (
            <h3 className="text-sm font-semibold text-destructive mb-1">{title}</h3>
          )}
          <p className="text-sm text-destructive">{message}</p>
        </div>
      </div>
    );
  }
);

ErrorMessage.displayName = 'ErrorMessage';

export { ErrorMessage };




