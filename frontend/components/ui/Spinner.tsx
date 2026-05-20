'use client';

import { HTMLAttributes } from 'react';

interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
  color?: 'primary' | 'white' | 'gray';
}

const sizeClasses = {
  sm: 'h-4 w-4 border-2',
  md: 'h-8 w-8 border-2',
  lg: 'h-12 w-12 border-4',
};

const colorClasses = {
  primary: 'border-primary-600 border-t-transparent',
  white: 'border-white border-t-transparent',
  gray: 'border-gray-600 border-t-transparent',
};

export function Spinner({
  size = 'md',
  color = 'primary',
  className = '',
  ...props
}: SpinnerProps) {
  return (
    <div
      className={`
        animate-spin
        rounded-full
        ${sizeClasses[size]}
        ${colorClasses[color]}
        ${className}
      `}
      role="status"
      aria-label="Cargando"
      {...props}
    >
      <span className="sr-only">Cargando...</span>
    </div>
  );
}

