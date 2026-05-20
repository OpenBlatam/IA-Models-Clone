'use client';

import React, { memo } from 'react';
import { AlertCircle, CheckCircle, Info, XCircle, X } from 'lucide-react';
import { clsx } from 'clsx';

interface AlertProps {
  children: React.ReactNode;
  variant?: 'success' | 'error' | 'warning' | 'info';
  title?: string;
  onClose?: () => void;
  className?: string;
}

const variants = {
  success: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-green-200 dark:border-green-800',
    text: 'text-green-800 dark:text-green-200',
    icon: CheckCircle,
    iconColor: 'text-green-600 dark:text-green-400',
  },
  error: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    border: 'border-red-200 dark:border-red-800',
    text: 'text-red-800 dark:text-red-200',
    icon: XCircle,
    iconColor: 'text-red-600 dark:text-red-400',
  },
  warning: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    border: 'border-yellow-200 dark:border-yellow-800',
    text: 'text-yellow-800 dark:text-yellow-200',
    icon: AlertCircle,
    iconColor: 'text-yellow-600 dark:text-yellow-400',
  },
  info: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-200 dark:border-blue-800',
    text: 'text-blue-800 dark:text-blue-200',
    icon: Info,
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
};

export const Alert: React.FC<AlertProps> = memo(({
  children,
  variant = 'info',
  title,
  onClose,
  className,
}) => {

  const variantStyles = variants[variant];
  const Icon = variantStyles.icon;

  return (
    <div
      className={clsx(
        'rounded-lg border p-4',
        variantStyles.bg,
        variantStyles.border,
        className
      )}
    >
      <div className="flex items-start">
        <Icon className={clsx('h-5 w-5 mt-0.5 mr-3', variantStyles.iconColor)} />
        <div className="flex-1">
          {title && (
            <h4 className={clsx('font-semibold mb-1', variantStyles.text)}>
              {title}
            </h4>
          )}
          <div className={variantStyles.text}>{children}</div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className={clsx(
              'ml-4 -mt-1 -mr-1 p-1 rounded-md',
              'hover:bg-opacity-20',
              variantStyles.text
            )}
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
});

Alert.displayName = 'Alert';

