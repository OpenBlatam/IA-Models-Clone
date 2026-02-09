'use client';

import { memo, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  headerActions?: ReactNode;
}

const Card = memo(({ children, className, title, headerActions }: CardProps): JSX.Element => {
  return (
    <div className={cn('bg-white rounded-lg shadow-md p-6', className)}>
      {title && (
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
          {headerActions && <div className="flex items-center space-x-2">{headerActions}</div>}
        </div>
      )}
      {children}
    </div>
  );
});

Card.displayName = 'Card';

export default Card;
