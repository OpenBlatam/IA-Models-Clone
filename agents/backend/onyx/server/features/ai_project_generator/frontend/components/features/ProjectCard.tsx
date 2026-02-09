'use client'

import clsx from 'clsx'
import type { Project } from '@/types'
import { formatDate, formatRelativeTime } from '@/lib/utils'
import { CheckCircle, XCircle, Clock, Loader, Download, FolderOpen } from 'lucide-react'
import { Badge, Button, Tooltip, CopyButton } from '@/components/ui'
import Card from '@/components/ui/Card'

interface ProjectCardProps {
  project: Project
  onExport?: (projectPath: string) => void
  onClick?: (project: Project) => void
}

const ProjectCard = ({ project, onExport, onClick }: ProjectCardProps) => {
  const getStatusIcon = () => {
    switch (project.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" aria-hidden="true" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" aria-hidden="true" />
      case 'processing':
        return <Loader className="w-5 h-5 text-blue-500 animate-spin" aria-hidden="true" />
      default:
        return <Clock className="w-5 h-5 text-yellow-500" aria-hidden="true" />
    }
  }

  const getStatusVariant = (): 'success' | 'danger' | 'warning' | 'info' => {
    switch (project.status) {
      case 'completed':
        return 'success'
      case 'failed':
        return 'danger'
      case 'processing':
        return 'info'
      default:
        return 'warning'
    }
  }

  const handleCardClick = () => {
    if (onClick) {
      onClick(project)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && onClick) {
      e.preventDefault()
      onClick(project)
    }
  }

  return (
    <Card
      hover={!!onClick}
      className={clsx(onClick && 'cursor-pointer')}
      onClick={handleCardClick}
      onKeyDown={handleKeyDown}
      tabIndex={onClick ? 0 : undefined}
      role={onClick ? 'button' : undefined}
      aria-label={onClick ? `View project ${project.project_name || project.id}` : undefined}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2">
            {getStatusIcon()}
            <h3 className="font-semibold text-gray-900 truncate">
              {project.project_name || project.id}
            </h3>
            <Badge variant={getStatusVariant()} size="sm">
              {project.status}
            </Badge>
          </div>

          <p className="text-gray-600 text-sm mb-3 line-clamp-2">{project.description}</p>

          <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500 mb-3">
            <span>Created: {formatRelativeTime(project.created_at)}</span>
            {project.completed_at && (
              <span>Completed: {formatRelativeTime(project.completed_at)}</span>
            )}
            {project.author && <span>Author: {project.author}</span>}
          </div>

          {project.error && (
            <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
              <strong>Error:</strong> {project.error}
            </div>
          )}

          {project.result && (
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-3">
              <FolderOpen className="w-4 h-4 flex-shrink-0" aria-hidden="true" />
              <span className="truncate flex-1">{project.result.project_dir}</span>
              <CopyButton
                text={project.result.project_dir}
                size="sm"
                variant="secondary"
                className="flex-shrink-0"
              />
            </div>
          )}
        </div>

        {project.result && onExport && (
          <div className="ml-4 flex-shrink-0">
            <Tooltip content="Export as ZIP" position="left">
              <Button
                variant="secondary"
                size="sm"
                leftIcon={<Download className="w-4 h-4" />}
                onClick={(e) => {
                  e.stopPropagation()
                  onExport(project.result!.project_dir)
                }}
                aria-label={`Export project ${project.project_name || project.id} as ZIP`}
              />
            </Tooltip>
          </div>
        )}
      </div>
    </Card>
  )
}

export default ProjectCard

