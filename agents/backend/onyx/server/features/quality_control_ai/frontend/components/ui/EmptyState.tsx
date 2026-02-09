'use client';

import { memo, type ReactNode } from 'react';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  className?: string;
  action?: ReactNode;
  size?: 'sm' | 'md' | 'lg';
}

const EmptyState = memo(
  ({
    icon: Icon,
    title,
    description,
    className,
    action,
    size = 'md',
  }: EmptyStateProps): JSX.Element => {
    const sizeClasses = {
      sm: { icon: 'w-8 h-8', title: 'text-sm', description: 'text-xs' },
      md: { icon: 'w-12 h-12', title: 'text-base', description: 'text-sm' },
      lg: { icon: 'w-16 h-16', title: 'text-lg', description: 'text-base' },
    };

    const sizes = sizeClasses[size];

    return (
      <div
        className={cn('text-center py-8 text-gray-500', className)}
        role="status"
        aria-live="polite"
        aria-label={`${title}${description ? `. ${description}` : ''}`}
      >
        {Icon && (
          <Icon
            className={cn('mx-auto mb-4 text-gray-400', sizes.icon)}
            aria-hidden="true"
          />
        )}
        <p className={cn('font-medium', sizes.title)}>{title}</p>
        {description && (
          <p className={cn('mt-1 text-gray-400', sizes.description)}>{description}</p>
        )}
        {action && <div className="mt-6">{action}</div>}
      </div>
    );
  }
);

EmptyState.displayName = 'EmptyState';

export default EmptyState;

