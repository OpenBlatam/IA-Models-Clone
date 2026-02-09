'use client'

import { useState, useMemo, useCallback } from 'react'
import clsx from 'clsx'
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react'

interface Column<T> {
  key: keyof T | string
  header: string
  render?: (value: unknown, row: T) => React.ReactNode
  sortable?: boolean
}

interface DataTableProps<T> {
  data: T[]
  columns: Column<T>[]
  onRowClick?: (row: T) => void
  className?: string
  emptyMessage?: string
}

const DataTable = <T extends Record<string, unknown>>({
  data,
  columns,
  onRowClick,
  className,
  emptyMessage = 'No data available',
}: DataTableProps<T>) => {
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null)

  const sortedData = useMemo(() => {
    if (!sortConfig) {
      return data
    }

    return [...data].sort((a, b) => {
      const aValue = a[sortConfig.key]
      const bValue = b[sortConfig.key]

      if (aValue === bValue) {
        return 0
      }

      const comparison = aValue < bValue ? -1 : 1
      return sortConfig.direction === 'asc' ? comparison : -comparison
    })
  }, [data, sortConfig])

  const handleSort = useCallback(
    (key: string) => {
      setSortConfig((prev) => {
        if (prev?.key === key) {
          return prev.direction === 'asc' ? { key, direction: 'desc' } : null
        }
        return { key, direction: 'asc' }
      })
    },
    []
  )

  const handleRowKeyDown = useCallback(
    (e: React.KeyboardEvent, row: T) => {
      if ((e.key === 'Enter' || e.key === ' ') && onRowClick) {
        e.preventDefault()
        onRowClick(row)
      }
    },
    [onRowClick]
  )

  if (data.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p>{emptyMessage}</p>
      </div>
    )
  }

  return (
    <div className={clsx('overflow-x-auto', className)}>
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((column) => (
              <th
                key={String(column.key)}
                className={clsx(
                  'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider',
                  column.sortable && 'cursor-pointer hover:bg-gray-100'
                )}
                onClick={() => column.sortable && handleSort(String(column.key))}
                onKeyDown={(e) => {
                  if ((e.key === 'Enter' || e.key === ' ') && column.sortable) {
                    e.preventDefault()
                    handleSort(String(column.key))
                  }
                }}
                tabIndex={column.sortable ? 0 : undefined}
                aria-sort={
                  column.sortable
                    ? sortConfig?.key === column.key
                      ? sortConfig.direction === 'asc'
                        ? 'ascending'
                        : 'descending'
                      : 'none'
                    : undefined
                }
              >
                <div className="flex items-center gap-2">
                  {column.header}
                  {column.sortable && (
                    <span className="flex-shrink-0">
                      {sortConfig?.key === column.key ? (
                        sortConfig.direction === 'asc' ? (
                          <ArrowUp className="w-4 h-4" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="w-4 h-4" aria-hidden="true" />
                        )
                      ) : (
                        <ArrowUpDown className="w-4 h-4 text-gray-400" aria-hidden="true" />
                      )}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedData.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              onClick={() => onRowClick?.(row)}
              onKeyDown={(e) => handleRowKeyDown(e, row)}
              className={clsx(
                'transition-colors',
                onRowClick && 'cursor-pointer hover:bg-gray-50'
              )}
              tabIndex={onRowClick ? 0 : undefined}
              role={onRowClick ? 'button' : undefined}
            >
              {columns.map((column) => (
                <td key={String(column.key)} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {column.render
                    ? column.render(row[column.key as keyof T], row)
                    : String(row[column.key as keyof T] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default DataTable

