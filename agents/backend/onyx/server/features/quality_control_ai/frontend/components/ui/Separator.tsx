'use client';

import { memo } from 'react';
import { cn } from '@/lib/utils';

interface SeparatorProps {
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

const Separator = memo(
  ({ orientation = 'horizontal', className }: SeparatorProps): JSX.Element => {
    return (
      <div
        className={cn(
          'bg-gray-200',
          orientation === 'horizontal' ? 'h-px w-full' : 'w-px h-full',
          className
        )}
        role="separator"
        aria-orientation={orientation}
      />
    );
  }
);

Separator.displayName = 'Separator';

export default Separator;

