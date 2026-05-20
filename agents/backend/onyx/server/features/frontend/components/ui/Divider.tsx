'use client';

import { HTMLAttributes } from 'react';

interface DividerProps extends HTMLAttributes<HTMLHRElement> {
  orientation?: 'horizontal' | 'vertical';
  spacing?: 'none' | 'sm' | 'md' | 'lg';
}

const spacingClasses = {
  none: '',
  sm: 'my-2',
  md: 'my-4',
  lg: 'my-6',
};

export function Divider({
  orientation = 'horizontal',
  spacing = 'md',
  className = '',
  ...props
}: DividerProps) {
  if (orientation === 'vertical') {
    return (
      <div
        className={`
          w-px
          h-full
          bg-gray-200 dark:bg-gray-700
          ${className}
        `}
        {...(props as any)}
      />
    );
  }

  return (
    <hr
      className={`
        border-0
        border-t
        border-gray-200 dark:border-gray-700
        ${spacingClasses[spacing]}
        ${className}
      `}
      {...props}
    />
  );
}

