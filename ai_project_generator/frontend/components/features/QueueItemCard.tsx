'use client'

import clsx from 'clsx'
import type { QueueItem } from '@/types'
import { formatDate, formatRelativeTime } from '@/lib/utils'
import { Trash2 } from 'lucide-react'
import { Badge, Button } from '@/components/ui'
import Card from '@/components/ui/Card'

interface QueueItemCardProps {
  item: QueueItem
  onDelete?: (projectId: string) => void
  isDeleting?: boolean
}

const QueueItemCard = ({ item, onDelete, isDeleting = false }: QueueItemCardProps) => {
  const getPriorityVariant = (): 'success' | 'warning' | 'info' => {
    if (item.priority > 0) {
      return 'success'
    }
    if (item.priority < 0) {
      return 'warning'
    }
    return 'info'
  }

  return (
    <Card hover className="transition-all">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <h3 className="font-semibold text-gray-900 truncate">
              {item.project_name || 'Unnamed Project'}
            </h3>
            <Badge variant={getPriorityVariant()} size="sm">
              Priority: {item.priority}
            </Badge>
            <Badge variant="warning" size="sm">
              {item.status}
            </Badge>
          </div>

          <p className="text-gray-600 text-sm mb-2 line-clamp-2">{item.description}</p>

          <p className="text-gray-400 text-xs">
            Created: {formatRelativeTime(item.created_at)} ({formatDate(item.created_at)})
          </p>
        </div>

        {onDelete && (
          <div className="ml-4 flex-shrink-0">
            <Button
              variant="danger"
              size="sm"
              leftIcon={<Trash2 className="w-5 h-5" />}
              onClick={() => onDelete(item.id)}
              disabled={isDeleting}
              isLoading={isDeleting}
              aria-label={`Delete project ${item.id}`}
            />
          </div>
        )}
      </div>
    </Card>
  )
}

export default QueueItemCard

