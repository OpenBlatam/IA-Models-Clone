/**
 * Data table component with sorting and pagination
 */

import React, { useState, useMemo } from 'react';
import { cn } from '@/lib/utils/cn';
import { Button } from './Button';
import { Pagination } from './Pagination';
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

export interface Column<T> {
  id: string;
  header: string;
  accessor: (row: T) => React.ReactNode;
  sortable?: boolean;
}

export interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  pageSize?: number;
  className?: string;
  emptyMessage?: string;
}

const DataTable = <T extends { id?: string }>({
  data,
  columns,
  pageSize = 10,
  className,
  emptyMessage = 'No hay datos disponibles',
}: DataTableProps<T>) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);

  const sortedData = useMemo(() => {
    if (!sortConfig) {
      return data;
    }

    return [...data].sort((a, b) => {
      const column = columns.find((col) => col.id === sortConfig.key);
      if (!column || !column.sortable) {
        return 0;
      }

      const aValue = column.accessor(a);
      const bValue = column.accessor(b);

      if (aValue === bValue) {
        return 0;
      }

      const comparison = aValue > bValue ? 1 : -1;
      return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
  }, [data, sortConfig, columns]);

  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    return sortedData.slice(startIndex, startIndex + pageSize);
  }, [sortedData, currentPage, pageSize]);

  const totalPages = Math.ceil(sortedData.length / pageSize);

  const handleSort = (columnId: string) => {
    const column = columns.find((col) => col.id === columnId);
    if (!column || !column.sortable) {
      return;
    }

    setSortConfig((prev) => {
      if (prev?.key === columnId) {
        return prev.direction === 'asc'
          ? { key: columnId, direction: 'desc' }
          : null;
      }
      return { key: columnId, direction: 'asc' };
    });
    setCurrentPage(1);
  };

  const handleKeyDown = (event: React.KeyboardEvent, columnId: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleSort(columnId);
    }
  };

  if (data.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">{emptyMessage}</div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      <div className="border rounded-lg overflow-hidden">
        <table className="w-full" role="table" aria-label="Tabla de datos">
          <thead className="bg-muted">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.id}
                  className="px-4 py-3 text-left text-sm font-medium text-foreground"
                  scope="col"
                >
                  {column.sortable ? (
                    <button
                      type="button"
                      onClick={() => handleSort(column.id)}
                      onKeyDown={(e) => handleKeyDown(e, column.id)}
                      className="flex items-center gap-2 hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      aria-label={`Ordenar por ${column.header}`}
                      tabIndex={0}
                    >
                      {column.header}
                      {sortConfig?.key === column.id ? (
                        sortConfig.direction === 'asc' ? (
                          <ArrowUp className="h-4 w-4" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="h-4 w-4" aria-hidden="true" />
                        )
                      ) : (
                        <ArrowUpDown className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                      )}
                    </button>
                  ) : (
                    column.header
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y">
            {paginatedData.map((row, index) => (
              <tr
                key={row.id || index}
                className="hover:bg-muted/50 transition-colors"
              >
                {columns.map((column) => (
                  <td key={column.id} className="px-4 py-3 text-sm">
                    {column.accessor(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={setCurrentPage}
        />
      )}
    </div>
  );
};

export { DataTable };




