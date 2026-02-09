'use client';

import { HTMLAttributes } from 'react';

interface ContainerProps extends HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  centered?: boolean;
}

const sizeClasses = {
  sm: 'max-w-2xl',
  md: 'max-w-4xl',
  lg: 'max-w-6xl',
  xl: 'max-w-7xl',
  full: 'max-w-full',
};

export function Container({
  children,
  size = 'lg',
  centered = true,
  className = '',
  ...props
}: ContainerProps) {
  return (
    <div
      className={`
        w-full
        ${sizeClasses[size]}
        ${centered ? 'mx-auto' : ''}
        px-4 sm:px-6 lg:px-8
        ${className}
      `}
      {...props}
    >
      {children}
    </div>
  );
}

