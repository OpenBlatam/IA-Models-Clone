'use client';

import { Inbox } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface EmptyTableProps {
  title?: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyTable = ({
  title = 'No hay datos',
  description = 'No se encontraron registros para mostrar',
  actionLabel,
  onAction,
  className,
}: EmptyTableProps) => {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12 px-4 text-center',
        className
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 mb-4">
        <Inbox className="h-6 w-6 text-gray-400 dark:text-gray-500" />
      </div>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">{title}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 max-w-sm mb-4">{description}</p>
      {actionLabel && onAction && (
        <Button onClick={onAction} variant="primary" size="sm">
          {actionLabel}
        </Button>
      )}
    </div>
  );
};



