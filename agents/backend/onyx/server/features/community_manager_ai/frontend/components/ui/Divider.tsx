'use client';

import { cn } from '@/lib/utils';

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  label?: string;
  className?: string;
}

export const Divider = ({ orientation = 'horizontal', label, className }: DividerProps) => {
  if (orientation === 'vertical') {
    return (
      <div
        className={cn('w-px bg-gray-200 dark:bg-gray-700 self-stretch', className)}
        role="separator"
        aria-orientation="vertical"
      />
    );
  }

  if (label) {
    return (
      <div className={cn('flex items-center gap-4', className)}>
        <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
        <span className="text-sm text-gray-500 dark:text-gray-400">{label}</span>
        <div className="flex-1 h-px bg-gray-200 dark:bg-gray-700" />
      </div>
    );
  }

  return (
    <div
      className={cn('h-px bg-gray-200 dark:bg-gray-700', className)}
      role="separator"
      aria-orientation="horizontal"
    />
  );
};



