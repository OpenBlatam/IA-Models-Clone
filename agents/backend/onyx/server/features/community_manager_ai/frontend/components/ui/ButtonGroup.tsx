'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface ButtonGroupProps {
  children: ReactNode;
  orientation?: 'horizontal' | 'vertical';
  className?: string;
  attached?: boolean;
}

export const ButtonGroup = ({
  children,
  orientation = 'horizontal',
  attached = false,
  className,
}: ButtonGroupProps) => {
  return (
    <div
      className={cn(
        'flex',
        orientation === 'horizontal' ? 'flex-row' : 'flex-col',
        attached && orientation === 'horizontal' && '[&>*:not(:first-child)]:ml-0 [&>*:not(:first-child)]:rounded-l-none [&>*:not(:last-child)]:rounded-r-none',
        attached && orientation === 'vertical' && '[&>*:not(:first-child)]:mt-0 [&>*:not(:first-child)]:rounded-t-none [&>*:not(:last-child)]:rounded-b-none',
        !attached && orientation === 'horizontal' && 'gap-2',
        !attached && orientation === 'vertical' && 'gap-2',
        className
      )}
      role="group"
    >
      {children}
    </div>
  );
};



