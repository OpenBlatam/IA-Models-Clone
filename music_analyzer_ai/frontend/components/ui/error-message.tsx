/**
 * Error message component.
 * Displays validation and error messages consistently.
 */

import { AlertCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ErrorMessageProps {
  message: string;
  className?: string;
  onDismiss?: () => void;
  variant?: 'default' | 'inline' | 'banner';
}

/**
 * Error message component for displaying validation and error messages.
 * @param props - Component props
 * @returns Error message component
 */
export function ErrorMessage({
  message,
  className,
  onDismiss,
  variant = 'default',
}: ErrorMessageProps) {
  if (variant === 'inline') {
    return (
      <span
        className={cn('text-sm text-red-300 flex items-center gap-1', className)}
        role="alert"
      >
        <AlertCircle className="w-4 h-4" />
        {message}
      </span>
    );
  }

  if (variant === 'banner') {
    return (
      <div
        className={cn(
          'bg-red-500/20 border border-red-500/50 rounded-lg p-4 flex items-center justify-between',
          className
        )}
        role="alert"
      >
        <div className="flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-300" />
          <span className="text-red-300">{message}</span>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-red-300 hover:text-red-200 transition-colors"
            aria-label="Dismiss error"
            type="button"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'bg-red-500/10 border border-red-500/30 rounded-lg p-3 flex items-center gap-2',
        className
      )}
      role="alert"
    >
      <AlertCircle className="w-5 h-5 text-red-300 flex-shrink-0" />
      <span className="text-red-300 text-sm">{message}</span>
    </div>
  );
}

interface FieldErrorProps {
  error?: string;
  className?: string;
}

/**
 * Field error component for form validation errors.
 * @param props - Component props
 * @returns Field error component
 */
export function FieldError({ error, className }: FieldErrorProps) {
  if (!error) return null;

  return (
    <ErrorMessage
      message={error}
      variant="inline"
      className={className}
    />
  );
}

