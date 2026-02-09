'use client';

import { cn } from '@/lib/utils';

interface DividerProps {
  label?: string;
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

const Divider = ({ label, orientation = 'horizontal', className }: DividerProps) => {
  if (orientation === 'vertical') {
    return <div className={cn('w-px bg-gray-200', className)} aria-orientation="vertical" />;
  }

  if (label) {
    return (
      <div className={cn('relative flex items-center py-4', className)}>
        <div className="flex-grow border-t border-gray-200" />
        <span className="px-4 text-sm text-gray-500">{label}</span>
        <div className="flex-grow border-t border-gray-200" />
      </div>
    );
  }

  return <div className={cn('border-t border-gray-200', className)} />;
};

export { Divider };

