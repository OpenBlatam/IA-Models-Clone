'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-9 h-5 after:h-4 after:w-4',
  md: 'w-11 h-6 after:h-5 after:w-5',
  lg: 'w-14 h-7 after:h-6 after:w-6',
};

export const Toggle: React.FC<ToggleProps> = memo(({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  size = 'md',
  className,
}) => {
  return (
    <div className={clsx('flex items-center justify-between', className)}>
      {(label || description) && (
        <div className="flex-1 mr-4">
          {label && (
            <p className="font-medium text-gray-900 dark:text-white">{label}</p>
          )}
          {description && (
            <p className="text-sm text-gray-500 dark:text-gray-400">{description}</p>
          )}
        </div>
      )}
      <label className="relative inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="sr-only peer"
        />
        <div
          className={clsx(
            'bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer',
            'peer-checked:after:translate-x-full peer-checked:after:border-white',
            'after:content-[""] after:absolute after:top-[2px] after:left-[2px]',
            'after:bg-white after:border-gray-300 after:border after:rounded-full',
            'after:transition-all peer-checked:bg-primary-600',
            'dark:bg-gray-700 dark:peer-checked:bg-primary-600',
            sizeClasses[size],
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        />
      </label>
    </div>
  );
});

Toggle.displayName = 'Toggle';



