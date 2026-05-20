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
}

interface TableHeadProps {
  children: ReactNode;
  className?: string;
}

interface TableCellProps {
  children: ReactNode;
  className?: string;
}

const Table = ({ children, className }: TableProps) => {
  return (
    <div className="overflow-x-auto">
      <table className={cn('w-full border-collapse', className)}>{children}</table>
    </div>
  );
};

const TableHeader = ({ children, className }: TableHeaderProps) => {
  return <thead className={cn('bg-gray-50', className)}>{children}</thead>;
};

const TableBody = ({ children, className }: TableBodyProps) => {
  return <tbody className={cn('bg-white divide-y divide-gray-200', className)}>{children}</tbody>;
};

const TableRow = ({ children, className, onClick }: TableRowProps) => {
  return (
    <tr
      className={cn(
        'hover:bg-gray-50 transition-colors',
        onClick && 'cursor-pointer',
        className
      )}
      onClick={onClick}
      onKeyDown={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick();
        }
      }}
      tabIndex={onClick ? 0 : undefined}
      role={onClick ? 'button' : undefined}
    >
      {children}
    </tr>
  );
};

const TableHead = ({ children, className }: TableHeadProps) => {
  return (
    <th className={cn('px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider', className)}>
      {children}
    </th>
  );
};

const TableCell = ({ children, className }: TableCellProps) => {
  return <td className={cn('px-6 py-4 whitespace-nowrap text-sm text-gray-900', className)}>{children}</td>;
};

export { Table, TableHeader, TableBody, TableRow, TableHead, TableCell };

