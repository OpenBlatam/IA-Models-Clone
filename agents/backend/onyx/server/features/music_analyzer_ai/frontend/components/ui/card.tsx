/**
 * Card component.
 * Reusable card component with header, body, and footer sections.
 */

import { type HTMLAttributes, forwardRef } from 'react';
import { cn } from '@/lib/utils';

/**
 * Card component.
 */
export const Card = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'bg-white/10 backdrop-blur-lg rounded-xl',
          'border border-white/20',
          'shadow-lg',
          className
        )}
        {...props}
      />
    );
  }
);
Card.displayName = 'Card';

/**
 * Card header component.
 */
export const CardHeader = forwardRef<
  HTMLDivElement,
  HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn('p-6 pb-4', className)}
      {...props}
    />
  );
});
CardHeader.displayName = 'CardHeader';

/**
 * Card title component.
 */
export const CardTitle = forwardRef<
  HTMLHeadingElement,
  HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => {
  return (
    <h3
      ref={ref}
      className={cn('text-xl font-bold text-white', className)}
      {...props}
    />
  );
});
CardTitle.displayName = 'CardTitle';

/**
 * Card description component.
 */
export const CardDescription = forwardRef<
  HTMLParagraphElement,
  HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => {
  return (
    <p
      ref={ref}
      className={cn('text-sm text-gray-400 mt-1', className)}
      {...props}
    />
  );
});
CardDescription.displayName = 'CardDescription';

/**
 * Card body component.
 */
export const CardContent = forwardRef<
  HTMLDivElement,
  HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn('p-6 pt-0', className)}
      {...props}
    />
  );
});
CardContent.displayName = 'CardContent';

/**
 * Card footer component.
 */
export const CardFooter = forwardRef<
  HTMLDivElement,
  HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn('p-6 pt-4 border-t border-white/10', className)}
      {...props}
    />
  );
});
CardFooter.displayName = 'CardFooter';

