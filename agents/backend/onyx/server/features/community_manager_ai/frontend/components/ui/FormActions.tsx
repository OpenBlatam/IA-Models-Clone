'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface FormActionsProps {
  children: ReactNode;
  className?: string;
  align?: 'left' | 'right' | 'center' | 'between';
}

const alignClasses = {
  left: 'justify-start',
  right: 'justify-end',
  center: 'justify-center',
  between: 'justify-between',
};

export const FormActions = ({
  children,
  className,
  align = 'right',
}: FormActionsProps) => {
  return (
    <div
      className={cn(
        'flex items-center gap-2',
        alignClasses[align],
        className
      )}
    >
      {children}
    </div>
  );
};



