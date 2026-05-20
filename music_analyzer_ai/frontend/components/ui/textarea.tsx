/**
 * Textarea component.
 * Reusable textarea component with validation states.
 */

import { type TextareaHTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

export interface TextareaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: boolean;
  fullWidth?: boolean;
}

/**
 * Textarea component.
 * Provides consistent textarea styling with validation states.
 *
 * @param props - Component props
 * @returns Textarea component
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, error = false, fullWidth = false, ...props }, ref) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          'px-4 py-2',
          'bg-white/10 border rounded-lg',
          'text-white placeholder-gray-400',
          'focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-slate-900',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          'resize-y min-h-[100px]',
          'transition-colors',
          error
            ? 'border-red-500 focus:ring-red-400'
            : 'border-white/20 focus:border-purple-400',
          fullWidth && 'w-full',
          className
        )}
        {...props}
      />
    );
  }
);
Textarea.displayName = 'Textarea';

