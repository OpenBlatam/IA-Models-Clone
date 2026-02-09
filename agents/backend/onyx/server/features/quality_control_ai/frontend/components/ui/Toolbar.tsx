'use client';

import { memo, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface ToolbarProps {
  children: ReactNode;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

const Toolbar = memo(
  ({ children, className, orientation = 'horizontal' }: ToolbarProps): JSX.Element => {
    return (
      <div
        className={cn(
          'flex items-center gap-2',
          orientation === 'vertical' && 'flex-col',
          className
        )}
        role="toolbar"
        aria-orientation={orientation}
      >
        {children}
      </div>
    );
  }
);

Toolbar.displayName = 'Toolbar';

export default Toolbar;

