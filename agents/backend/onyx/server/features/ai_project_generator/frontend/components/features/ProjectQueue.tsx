'use client'

import { useState, useCallback } from 'react'
import { api } from '@/lib/api'
import { getErrorMessage } from '@/lib/utils'
import type { QueueResponse } from '@/types'
import { Clock, RefreshCw } from 'lucide-react'
import { Button, ErrorMessage, EmptyState, ConfirmDialog, LoadingSpinner, Card } from '@/components/ui'
import QueueItemCard from './QueueItemCard'

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

  if (!queue) {
    return (
      <Card>
        <LoadingSpinner size="lg" text="Loading queue..." />
      </Card>
    )
  }

  if (queue.queue_size === 0) {
    return (
      <Card>
        <EmptyState
          icon={Clock}
          title="Queue is empty"
          description="Generate a new project to see it here"
        />
      </Card>
    )
  }

  return (
    <>
      <Card>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">
            Project Queue ({queue.queue_size})
          </h2>
          <Button
            variant="secondary"
            size="md"
            leftIcon={<RefreshCw className="w-4 h-4" />}
            onClick={onRefresh}
            aria-label="Refresh Queue"
          >
            Refresh
          </Button>
        </div>

        {error && (
          <div className="mb-4">
            <ErrorMessage message={error} onDismiss={() => setError(null)} />
          </div>
        )}

        <div className="space-y-4">
          {queue.queue.map((item) => (
            <QueueItemCard
              key={item.id}
              item={item}
              onDelete={() => handleDeleteClick(item.id, item.project_name || 'Unnamed Project')}
              isDeleting={deleting === item.id}
            />
          ))}
        </div>
      </Card>

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

