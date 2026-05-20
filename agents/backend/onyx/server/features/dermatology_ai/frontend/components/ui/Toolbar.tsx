'use client';

import React from 'react';
import { clsx } from 'clsx';

interface ToolbarItem {
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}

interface ToolbarProps {
  items: ToolbarItem[];
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

export const Toolbar: React.FC<ToolbarProps> = ({
  items,
  orientation = 'horizontal',
  className,
}) => {
  return (
    <div
      className={clsx(
        'flex bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-1',
        orientation === 'horizontal' ? 'flex-row' : 'flex-col',
        className
      )}
      role="toolbar"
      aria-label="Barra de herramientas"
    >
      {items.map((item, index) => (
        <button
          key={index}
          onClick={item.onClick}
          disabled={item.disabled}
          className={clsx(
            'px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2',
            item.active
              ? 'bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300'
              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700',
            item.disabled && 'opacity-50 cursor-not-allowed'
          )}
          aria-label={item.label}
        >
          {item.icon && <span>{item.icon}</span>}
          <span>{item.label}</span>
        </button>
      ))}
    </div>
  );
};


