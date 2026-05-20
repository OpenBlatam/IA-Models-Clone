'use client';

import React, { memo } from 'react';
import { clsx } from 'clsx';

interface TabButtonProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
  icon?: React.ReactNode;
  className?: string;
}

export const TabButton: React.FC<TabButtonProps> = memo(({
  label,
  isActive,
  onClick,
  icon,
  className,
}) => {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'px-6 py-3 font-semibold rounded-lg transition-all duration-300 relative overflow-hidden group',
        isActive
          ? 'text-blue-600 dark:text-blue-400 bg-white dark:bg-gray-800 shadow-lg scale-105'
          : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-white/70 dark:hover:bg-gray-800/70',
        className
      )}
    >
      {isActive && (
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10" />
      )}
      <span className="relative z-10 flex items-center gap-2">
        {icon && <span>{icon}</span>}
        {label}
      </span>
    </button>
  );
});

TabButton.displayName = 'TabButton';



