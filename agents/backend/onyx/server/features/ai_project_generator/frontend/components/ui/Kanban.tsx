'use client'

import { useState, useCallback, ReactNode } from 'react'
import { Plus, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Card } from '@/components/ui'

interface KanbanItem {
  id: string
  title: string
  description?: string
  content?: ReactNode
}

interface KanbanColumn {
  id: string
  title: string
  items: KanbanItem[]
}

interface KanbanProps {
  columns: KanbanColumn[]
  onItemMove?: (itemId: string, fromColumnId: string, toColumnId: string) => void
  onItemAdd?: (columnId: string, item: Omit<KanbanItem, 'id'>) => void
  onItemRemove?: (itemId: string, columnId: string) => void
  className?: string
  allowAdd?: boolean
}

const Kanban = ({
  columns: initialColumns,
  onItemMove,
  onItemAdd,
  onItemRemove,
  className,
  allowAdd = true,
}: KanbanProps) => {
  const [columns, setColumns] = useState(initialColumns)
  const [draggedItem, setDraggedItem] = useState<{ itemId: string; columnId: string } | null>(null)

  const handleDragStart = useCallback((itemId: string, columnId: string) => {
    setDraggedItem({ itemId, columnId })
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent, targetColumnId: string) => {
      e.preventDefault()
      if (!draggedItem) {
        return
      }

      const { itemId, columnId: sourceColumnId } = draggedItem

      if (sourceColumnId === targetColumnId) {
        setDraggedItem(null)
        return
      }

      const sourceColumn = columns.find((col) => col.id === sourceColumnId)
      const targetColumn = columns.find((col) => col.id === targetColumnId)

      if (!sourceColumn || !targetColumn) {
        setDraggedItem(null)
        return
      }

      const item = sourceColumn.items.find((item) => item.id === itemId)
      if (!item) {
        setDraggedItem(null)
        return
      }

      const newColumns = columns.map((col) => {
        if (col.id === sourceColumnId) {
          return {
            ...col,
            items: col.items.filter((item) => item.id !== itemId),
          }
        }
        if (col.id === targetColumnId) {
          return {
            ...col,
            items: [...col.items, item],
          }
        }
        return col
      })

      setColumns(newColumns)
      onItemMove?.(itemId, sourceColumnId, targetColumnId)
      setDraggedItem(null)
    },
    [draggedItem, columns, onItemMove]
  )

  const handleRemove = useCallback(
    (itemId: string, columnId: string) => {
      const newColumns = columns.map((col) => {
        if (col.id === columnId) {
          return {
            ...col,
            items: col.items.filter((item) => item.id !== itemId),
          }
        }
        return col
      })
      setColumns(newColumns)
      onItemRemove?.(itemId, columnId)
    },
    [columns, onItemRemove]
  )

  return (
    <div className={cn('flex gap-4 overflow-x-auto pb-4', className)}>
      {columns.map((column) => (
        <div
          key={column.id}
          className="flex-shrink-0 w-80"
          onDragOver={handleDragOver}
          onDrop={(e) => handleDrop(e, column.id)}
        >
          <Card className="h-full flex flex-col">
            <div className="p-4 border-b border-gray-200">
              <h3 className="font-semibold text-gray-900">{column.title}</h3>
              <span className="text-sm text-gray-500">{column.items.length} items</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {column.items.map((item) => (
                <div
                  key={item.id}
                  draggable
                  onDragStart={() => handleDragStart(item.id, column.id)}
                  className="bg-white border border-gray-200 rounded-lg p-3 cursor-move hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-gray-900 mb-1">{item.title}</h4>
                      {item.description && (
                        <p className="text-sm text-gray-600">{item.description}</p>
                      )}
                      {item.content}
                    </div>
                    {onItemRemove && (
                      <button
                        onClick={() => handleRemove(item.id, column.id)}
                        className="flex-shrink-0 p-1 hover:bg-gray-100 rounded"
                        aria-label="Remove item"
                      >
                        <X className="w-4 h-4 text-gray-400" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {allowAdd && onItemAdd && (
              <div className="p-4 border-t border-gray-200">
                <button
                  onClick={() => onItemAdd(column.id, { title: 'New Item' })}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 rounded-lg border border-dashed border-gray-300"
                >
                  <Plus className="w-4 h-4" />
                  Add item
                </button>
              </div>
            )}
          </Card>
        </div>
      ))}
    </div>
  )
}

export default Kanban

