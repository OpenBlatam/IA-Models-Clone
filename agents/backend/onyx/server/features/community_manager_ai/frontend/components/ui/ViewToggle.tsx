'use client';

import { LayoutGrid, List } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/lib/utils';

type ViewMode = 'grid' | 'list';

interface ViewToggleProps {
  value: ViewMode;
  onChange: (mode: ViewMode) => void;
  className?: string;
}

export const ViewToggle = ({ value, onChange, className }: ViewToggleProps) => {
  return (
    <div className={cn('flex items-center gap-1 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-1', className)}>
      <button
        type="button"
        onClick={() => onChange('grid')}
        className={cn(
          'flex items-center justify-center rounded-md p-2 transition-colors',
          value === 'grid'
            ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400'
            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
        )}
        aria-label="Vista de cuadrícula"
        aria-pressed={value === 'grid'}
      >
        <LayoutGrid className="h-4 w-4" />
      </button>
      <button
        type="button"
        onClick={() => onChange('list')}
        className={cn(
          'flex items-center justify-center rounded-md p-2 transition-colors',
          value === 'list'
            ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400'
            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
        )}
        aria-label="Vista de lista"
        aria-pressed={value === 'list'}
      >
        <List className="h-4 w-4" />
      </button>
    </div>
  );
};



