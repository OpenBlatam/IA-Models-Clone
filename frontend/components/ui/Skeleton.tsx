'use client';

import { HTMLAttributes } from 'react';

interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

const variantClasses = {
  text: 'rounded',
  circular: 'rounded-full',
  rectangular: 'rounded-lg',
};

const animationClasses = {
  pulse: 'animate-pulse',
  wave: 'animate-pulse',
  none: '',
};

export function Skeleton({
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse',
  className = '',
  style,
  ...props
}: SkeletonProps) {
  return (
    <div
      className={`
        bg-gray-200 dark:bg-gray-700
        ${variantClasses[variant]}
        ${animationClasses[animation]}
        ${className}
      `}
      style={{
        width: width || '100%',
        height: height || '1rem',
        ...style,
      }}
      {...props}
    />
  );
}

