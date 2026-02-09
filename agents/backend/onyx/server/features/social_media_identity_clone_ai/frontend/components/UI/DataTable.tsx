import { cn } from '@/lib/utils';

interface Column<T> {
  key: keyof T | string;
  label: string;
  render?: (value: unknown, row: T) => React.ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  keyExtractor: (row: T) => string;
  className?: string;
  onRowClick?: (row: T) => void;
}

const DataTable = <T extends Record<string, unknown>>({
  data,
  columns,
  keyExtractor,
  className = '',
  onRowClick,
}: DataTableProps<T>): JSX.Element => {
  const handleRowClick = (row: T): void => {
    if (onRowClick) {
      onRowClick(row);
    }
  };

  const handleRowKeyDown = (e: React.KeyboardEvent<HTMLTableRowElement>, row: T): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleRowClick(row);
    }
  };

  if (data.length === 0) {
    return (
      <div className={cn('text-center py-8 text-gray-600', className)} role="status" aria-live="polite">
        No data available
      </div>
    );
  }

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-gray-200">
            {columns.map((column) => (
              <th
                key={String(column.key)}
                className={cn('px-4 py-3 text-left text-sm font-semibold text-gray-700', column.className)}
                scope="col"
              >
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const rowKey = keyExtractor(row);
            const isClickable = Boolean(onRowClick);

            return (
              <tr
                key={rowKey}
                onClick={() => handleRowClick(row)}
                onKeyDown={(e) => handleRowKeyDown(e, row)}
                className={cn(
                  'border-b border-gray-100 hover:bg-gray-50 transition-colors',
                  isClickable && 'cursor-pointer'
                )}
                role={isClickable ? 'button' : 'row'}
                tabIndex={isClickable ? 0 : undefined}
                aria-label={isClickable ? `View details for row ${rowKey}` : undefined}
              >
                {columns.map((column) => {
                  const cellKey = `${rowKey}-${String(column.key)}`;
                  const cellValue = row[column.key as keyof T];

                  return (
                    <td key={cellKey} className={cn('px-4 py-3 text-sm text-gray-700', column.className)}>
                      {column.render ? column.render(cellValue, row) : String(cellValue ?? '')}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default DataTable;



