'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface ButtonGroupProps {
  children: React.ReactNode;
  orientation?: 'horizontal' | 'vertical';
  spacing?: 'none' | 'sm' | 'md' | 'lg';
  className?: string;
}

const spacingClasses = {
  none: 'gap-0',
  sm: 'gap-2',
  md: 'gap-3',
  lg: 'gap-4',
};

export const ButtonGroup: React.FC<ButtonGroupProps> = memo(({
  children,
  orientation = 'horizontal',
  spacing = 'md',
  className,
}) => {
  return (
    <div
      className={clsx(
        'flex',
        orientation === 'horizontal' ? 'flex-row' : 'flex-col',
        spacingClasses[spacing],
        className
      )}
    >
      {children}
    </div>
  );
});

ButtonGroup.displayName = 'ButtonGroup';



