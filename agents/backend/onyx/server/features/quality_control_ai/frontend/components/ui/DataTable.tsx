'use client';

import { memo, useMemo, type ReactNode } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './Button';
import { Skeleton } from './Skeleton';

export type SortDirection = 'asc' | 'desc' | null;

interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (value: unknown, row: T) => ReactNode;
  sortable?: boolean;
  width?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  onSort?: (key: keyof T | string, direction: SortDirection) => void;
  sortKey?: keyof T | string | null;
  sortDirection?: SortDirection;
  emptyMessage?: string;
  className?: string;
  rowClassName?: (row: T, index: number) => string;
  onRowClick?: (row: T) => void;
}

const DataTable = memo(
  <T extends Record<string, unknown>>({
    data,
    columns,
    loading = false,
    onSort,
    sortKey,
    sortDirection,
    emptyMessage = 'No data available',
    className,
    rowClassName,
    onRowClick,
  }: DataTableProps<T>): JSX.Element => {
    const handleSort = (column: Column<T>): void => {
      if (!column.sortable || !onSort) return;

      const key = column.key;
      let newDirection: SortDirection = 'asc';

      if (sortKey === key) {
        if (sortDirection === 'asc') {
          newDirection = 'desc';
        } else if (sortDirection === 'desc') {
          newDirection = null;
        }
      }

      onSort(key, newDirection);
    };

    const getSortIcon = (column: Column<T>): ReactNode => {
      if (!column.sortable || sortKey !== column.key) {
        return null;
      }

      if (sortDirection === 'asc') {
        return <ChevronUp className="w-4 h-4 ml-1" aria-hidden="true" />;
      }

      if (sortDirection === 'desc') {
        return <ChevronDown className="w-4 h-4 ml-1" aria-hidden="true" />;
      }

      return null;
    };

    if (loading) {
      return (
        <div className={cn('overflow-x-auto', className)}>
          <table className="w-full border-collapse">
            <thead>
              <tr>
                {columns.map((column, index) => (
                  <th
                    key={index}
                    className="px-4 py-3 text-left text-sm font-semibold text-gray-700 bg-gray-50 border-b"
                    style={{ width: column.width }}
                  >
                    <Skeleton className="h-4 w-20" />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...Array(5)].map((_, rowIndex) => (
                <tr key={rowIndex}>
                  {columns.map((_, colIndex) => (
                    <td key={colIndex} className="px-4 py-3 border-b">
                      <Skeleton className="h-4 w-full" />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    if (data.length === 0) {
      return (
        <div className={cn('text-center py-12', className)}>
          <p className="text-gray-500">{emptyMessage}</p>
        </div>
      );
    }

    return (
      <div className={cn('overflow-x-auto', className)}>
        <table className="w-full border-collapse">
          <thead>
            <tr>
              {columns.map((column, index) => (
                <th
                  key={index}
                  className={cn(
                    'px-4 py-3 text-left text-sm font-semibold text-gray-700 bg-gray-50 border-b',
                    column.sortable && onSort && 'cursor-pointer hover:bg-gray-100 transition-colors'
                  )}
                  style={{ width: column.width }}
                  onClick={() => handleSort(column)}
                  role={column.sortable ? 'button' : undefined}
                  tabIndex={column.sortable ? 0 : undefined}
                  aria-sort={
                    sortKey === column.key
                      ? sortDirection === 'asc'
                        ? 'ascending'
                        : sortDirection === 'desc'
                        ? 'descending'
                        : 'none'
                      : undefined
                  }
                >
                  <div className="flex items-center">
                    {column.header}
                    {getSortIcon(column)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className={cn(
                  'hover:bg-gray-50 transition-colors',
                  onRowClick && 'cursor-pointer',
                  rowClassName?.(row, rowIndex)
                )}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((column, colIndex) => {
                  const value = (row[column.key as keyof T] as unknown) ?? '';
                  const renderedValue = column.render ? column.render(value, row) : value;

                  return (
                    <td key={colIndex} className="px-4 py-3 border-b text-sm text-gray-700">
                      {renderedValue}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
) as <T extends Record<string, unknown>>(props: DataTableProps<T>) => JSX.Element;

DataTable.displayName = 'DataTable';

export default DataTable;

