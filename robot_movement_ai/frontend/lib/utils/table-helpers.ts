/**
 * Helper functions for table components
 */

export interface TableColumn<T> {
  key: keyof T;
  label: string;
  sortable?: boolean;
  render?: (value: any, row: T) => React.ReactNode;
  width?: string | number;
  align?: 'left' | 'center' | 'right';
}

/**
 * Generate table columns from data
 */
export function generateColumns<T extends Record<string, any>>(
  data: T[],
  excludeKeys: string[] = []
): TableColumn<T>[] {
  if (data.length === 0) return [];

  const keys = Object.keys(data[0]).filter((key) => !excludeKeys.includes(key));

  return keys.map((key) => ({
    key: key as keyof T,
    label: key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase()),
    sortable: true,
  }));
}

/**
 * Format cell value
 */
export function formatCellValue(value: any): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/**
 * Get column width
 */
export function getColumnWidth(column: TableColumn<any>, totalWidth: number): number {
  if (typeof column.width === 'number') return column.width;
  if (typeof column.width === 'string' && column.width.endsWith('%')) {
    return (totalWidth * parseFloat(column.width)) / 100;
  }
  return totalWidth / 10; // Default width
}



