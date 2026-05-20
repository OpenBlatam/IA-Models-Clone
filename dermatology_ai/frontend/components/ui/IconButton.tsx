'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';
import { LucideIcon, Loader2 } from 'lucide-react';

interface IconButtonProps extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  icon: LucideIcon;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  isLoading?: boolean;
  'aria-label': string;
}

const sizeClasses = {
  sm: 'p-1.5',
  md: 'p-2',
  lg: 'p-3',
};

const iconSizeClasses = {
  sm: 'h-4 w-4',
  md: 'h-5 w-5',
  lg: 'h-6 w-6',
};

const variantClasses = {
  primary: 'bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500',
  secondary: 'bg-secondary-600 text-white hover:bg-secondary-700 focus:ring-secondary-500',
  outline: 'border-2 border-primary-600 text-primary-600 hover:bg-primary-50 focus:ring-primary-500',
  ghost: 'text-gray-700 hover:bg-gray-100 focus:ring-gray-500 dark:text-gray-300 dark:hover:bg-gray-800',
  danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
};

export const IconButton: React.FC<IconButtonProps> = memo(({
  icon: Icon,
  size = 'md',
  variant = 'ghost',
  isLoading = false,
  className,
  disabled,
  ...props
}) => {
  return (
    <button
      className={clsx(
        'inline-flex items-center justify-center rounded-lg transition-all duration-200',
        'focus:outline-none focus:ring-2 focus:ring-offset-2',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        sizeClasses[size],
        variantClasses[variant],
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <Loader2 className={clsx('animate-spin', iconSizeClasses[size])} />
      ) : (
        <Icon className={iconSizeClasses[size]} />
      )}
    </button>
  );
});

IconButton.displayName = 'IconButton';

