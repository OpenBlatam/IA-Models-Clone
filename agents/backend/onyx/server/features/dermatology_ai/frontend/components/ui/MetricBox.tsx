'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface MetricBoxProps {
  label: string;
  value: string | number;
  subtitle?: string;
  className?: string;
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger';
}

const variantClasses = {
  default: 'bg-gray-50 dark:bg-gray-800',
  primary: 'bg-primary-50 dark:bg-primary-900/20',
  success: 'bg-green-50 dark:bg-green-900/20',
  warning: 'bg-yellow-50 dark:bg-yellow-900/20',
  danger: 'bg-red-50 dark:bg-red-900/20',
};

const valueColorClasses = {
  default: 'text-gray-900 dark:text-white',
  primary: 'text-primary-600 dark:text-primary-400',
  success: 'text-green-600 dark:text-green-400',
  warning: 'text-yellow-600 dark:text-yellow-400',
  danger: 'text-red-600 dark:text-red-400',
};

export const MetricBox: React.FC<MetricBoxProps> = memo(({
  label,
  value,
  subtitle,
  className,
  variant = 'default',
}) => {
  return (
    <div
      className={clsx(
        'text-center p-4 rounded-lg',
        variantClasses[variant],
        className
      )}
    >
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{label}</p>
      <p
        className={clsx(
          'text-2xl font-bold',
          variant !== 'default' ? valueColorClasses[variant] : valueColorClasses.default
        )}
      >
        {value}
      </p>
      {subtitle && (
        <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">{subtitle}</p>
      )}
    </div>
  );
});

MetricBox.displayName = 'MetricBox';



