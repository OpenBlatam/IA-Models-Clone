'use client';

import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface TableProps {
  children: ReactNode;
  className?: string;
}

interface TableHeaderProps {
  children: ReactNode;
  className?: string;
}

interface TableBodyProps {
  children: ReactNode;
  className?: string;
}

interface TableRowProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  selected?: boolean;
}

interface TableHeadProps {
  children: ReactNode;
  className?: string;
  align?: 'left' | 'center' | 'right';
}

interface TableCellProps {
  children: ReactNode;
  className?: string;
  align?: 'left' | 'center' | 'right';
}

export const Table = ({ children, className }: TableProps) => {
  return (
    <div className="w-full overflow-auto">
      <table className={cn('w-full border-collapse', className)}>{children}</table>
    </div>
  );
};

export const TableHeader = ({ children, className }: TableHeaderProps) => {
  return (
    <thead className={cn('bg-gray-50 dark:bg-gray-800', className)}>{children}</thead>
  );
};

export const TableBody = ({ children, className }: TableBodyProps) => {
  return <tbody className={className}>{children}</tbody>;
};

export const TableRow = ({ children, className, onClick, selected }: TableRowProps) => {
  return (
    <tr
      onClick={onClick}
      className={cn(
        'border-b border-gray-200 dark:border-gray-700 transition-colors',
        onClick && 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800',
        selected && 'bg-primary-50 dark:bg-primary-900/20',
        className
      )}
    >
      {children}
    </tr>
  );
};

export const TableHead = ({ children, className, align = 'left' }: TableHeadProps) => {
  const alignClasses = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right',
  };

  return (
    <th
      className={cn(
        'px-4 py-3 text-left text-sm font-semibold text-gray-900 dark:text-gray-100',
        alignClasses[align],
        className
      )}
    >
      {children}
    </th>
  );
};

export const TableCell = ({ children, className, align = 'left' }: TableCellProps) => {
  const alignClasses = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right',
  };

  return (
    <td
      className={cn(
        'px-4 py-3 text-sm text-gray-700 dark:text-gray-300',
        alignClasses[align],
        className
      )}
    >
      {children}
    </td>
  );
};



