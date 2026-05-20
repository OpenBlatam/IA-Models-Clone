'use client'

import { useState, useCallback } from 'react'
import clsx from 'clsx'
import { api } from '@/lib/api'
import { getErrorMessage, formatDate } from '@/lib/utils'
import type { QueueResponse, QueueItem } from '@/types'
import { Clock, Trash2, RefreshCw } from 'lucide-react'
import ErrorMessage from './ErrorMessage'
import EmptyState from './EmptyState'
import ConfirmDialog from './ConfirmDialog'

interface ProjectQueueProps {
  queue: QueueResponse | null
  onRefresh: () => void
}

const ProjectQueue = ({ queue, onRefresh }: ProjectQueueProps) => {
  const [deleting, setDeleting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<{ id: string; name: string } | null>(null)

  const handleDeleteClick = useCallback((projectId: string, projectName: string) => {
    setConfirmDelete({ id: projectId, name: projectName })
  }, [])

  const handleDeleteConfirm = useCallback(async () => {
    if (!confirmDelete) {
      return
    }

    const projectId = confirmDelete.id
    setDeleting(projectId)
    setError(null)
    setConfirmDelete(null)

    try {
      await api.deleteProject(projectId)
      onRefresh()
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage || 'Failed to delete project')
      console.error('Error deleting project:', err)
    } finally {
      setDeleting(null)
    }
  }, [confirmDelete, onRefresh])

  const handleDeleteCancel = useCallback(() => {
    setConfirmDelete(null)
  }, [])

  const handleKeyDown = useCallback((e: React.KeyboardEvent, action: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action()
    }
  }, [])

  if (!queue) {
    return (
      <div className="card">
        <p className="text-gray-500">Loading queue...</p>
      </div>
    )
  }

  if (queue.queue_size === 0) {
    return (
      <div className="card">
        <EmptyState
          icon={Clock}
          title="Queue is empty"
          description="Generate a new project to see it here"
        />
      </div>
    )
  }

  return (
    <>
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">
            Project Queue ({queue.queue_size})
          </h2>
          <button
            onClick={onRefresh}
            className="btn btn-secondary flex items-center gap-2"
            tabIndex={0}
            aria-label="Refresh Queue"
            onKeyDown={(e) => handleKeyDown(e, onRefresh)}
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>

        {error && (
          <div className="mb-4">
            <ErrorMessage message={error} onDismiss={() => setError(null)} />
          </div>
        )}

        <div className="space-y-4">
        {queue.queue.map((item: QueueItem) => (
          <div
            key={item.id}
            className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="font-semibold text-gray-900">
                    {item.project_name || 'Unnamed Project'}
                  </h3>
                  <span
                    className={clsx(
                      'px-2 py-1 rounded text-xs font-medium',
                      item.priority > 0 && 'bg-green-100 text-green-800',
                      item.priority < 0 && 'bg-gray-100 text-gray-800',
                      item.priority === 0 && 'bg-blue-100 text-blue-800'
                    )}
                  >
                    Priority: {item.priority}
                  </span>
                  <span className="px-2 py-1 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                    {item.status}
                  </span>
                </div>
                <p className="text-gray-600 text-sm mb-2">{item.description}</p>
                <p className="text-gray-400 text-xs">
                  Created: {formatDate(item.created_at)}
                </p>
              </div>
              <button
                onClick={() => handleDeleteClick(item.id, item.project_name || 'Unnamed Project')}
                disabled={deleting === item.id}
                className={clsx(
                  'ml-4 p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-red-500',
                  deleting === item.id && 'opacity-50 cursor-not-allowed'
                )}
                tabIndex={0}
                aria-label={`Delete project ${item.id}`}
                aria-disabled={deleting === item.id}
                onKeyDown={(e) =>
                  handleKeyDown(e, () => handleDeleteClick(item.id, item.project_name || 'Unnamed Project'))
                }
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))}
        </div>
      </div>

      <ConfirmDialog
        isOpen={confirmDelete !== null}
        title="Remove Project from Queue"
        message={`Are you sure you want to remove "${confirmDelete?.name}" from the queue? This action cannot be undone.`}
        confirmLabel="Remove"
        cancelLabel="Cancel"
        variant="warning"
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
      />
    </>
  )
}

export default ProjectQueue

