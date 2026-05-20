'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface FilterBarProps {
  children: ReactNode;
  className?: string;
}

const FilterBar = ({ children, className }: FilterBarProps) => {
  return (
    <div className={cn('flex flex-wrap items-center gap-4 mb-6 p-4 bg-white rounded-lg shadow-sm', className)}>
      {children}
    </div>
  );
};

export { FilterBar };

