'use client'

import { useState, useEffect, useCallback } from 'react'
import clsx from 'clsx'
import { api } from '@/lib/api'
import { getErrorMessage, formatDate } from '@/lib/utils'
import type { Project } from '@/types'
import { Folder, Download, CheckCircle, XCircle, Clock, Loader } from 'lucide-react'
import ErrorMessage from './ErrorMessage'
import LoadingSpinner from './LoadingSpinner'
import EmptyState from './EmptyState'
import Tooltip from './Tooltip'

type FilterType = 'all' | 'completed' | 'failed' | 'processing'

const ProjectList = () => {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterType>('all')

  const loadProjects = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const params: { status?: string } = {}
      if (filter !== 'all') {
        params.status = filter
      }
      const data = await api.getProjects(params)
      setProjects(data.projects)
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage || 'Failed to load projects')
      console.error('Error loading projects:', err)
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => {
    loadProjects()
    const interval = setInterval(loadProjects, 5000)
    return () => clearInterval(interval)
  }, [loadProjects])

  const handleExport = useCallback(async (projectPath: string, format: 'zip' | 'tar') => {
    try {
      const response =
        format === 'zip'
          ? await api.exportZip(projectPath)
          : await api.exportTar(projectPath)
      const exportPath = response.zip_path || response.tar_path
      alert(`Project exported to: ${exportPath}`)
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      alert(errorMessage || 'Failed to export project')
      console.error('Error exporting project:', err)
    }
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'processing':
        return <Loader className="w-5 h-5 text-blue-500 animate-spin" />
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />
    }
  }

  const handleKeyDown = useCallback((e: React.KeyboardEvent, action: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      action()
    }
  }, [])

  const handleFilterChange = useCallback((newFilter: FilterType) => {
    setFilter(newFilter)
    setError(null)
  }, [])

  if (loading && projects.length === 0) {
    return (
      <div className="card">
        <LoadingSpinner size="lg" text="Loading projects..." />
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Projects</h2>
        <div className="flex gap-2" role="group" aria-label="Filter projects">
          {(['all', 'completed', 'failed', 'processing'] as const).map((f) => (
            <button
              key={f}
              onClick={() => handleFilterChange(f)}
              className={clsx(
                'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                filter === f
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              )}
              tabIndex={0}
              aria-label={`Filter by ${f}`}
              onKeyDown={(e) => handleKeyDown(e, () => handleFilterChange(f))}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mb-4">
          <ErrorMessage message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {projects.length === 0 ? (
        <EmptyState
          icon={Folder}
          title="No projects found"
          description={
            filter === 'all'
              ? 'No projects have been generated yet. Create your first project to get started!'
              : `No ${filter} projects found. Try a different filter.`
          }
        />
      ) : (
        <div className="space-y-4">
          {projects.map((project) => (
            <div
              key={project.id}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    {getStatusIcon(project.status)}
                    <h3 className="font-semibold text-gray-900">
                      {project.project_name || project.id}
                    </h3>
                    <span
                      className={clsx(
                        'px-2 py-1 rounded text-xs font-medium',
                        project.status === 'completed' && 'bg-green-100 text-green-800',
                        project.status === 'failed' && 'bg-red-100 text-red-800',
                        project.status === 'processing' && 'bg-blue-100 text-blue-800',
                        project.status !== 'completed' &&
                          project.status !== 'failed' &&
                          project.status !== 'processing' &&
                          'bg-yellow-100 text-yellow-800'
                      )}
                    >
                      {project.status}
                    </span>
                  </div>
                  <p className="text-gray-600 text-sm mb-2">{project.description}</p>
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span>Created: {formatDate(project.created_at)}</span>
                    {project.completed_at && (
                      <span>Completed: {formatDate(project.completed_at)}</span>
                    )}
                    {project.author && <span>Author: {project.author}</span>}
                  </div>
                  {project.error && (
                    <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                      Error: {project.error}
                    </div>
                  )}
                  {project.result && (
                    <div className="mt-2 text-sm text-gray-600">
                      <p>Path: {project.result.project_dir}</p>
                    </div>
                  )}
                </div>
                {project.result && (
                  <div className="ml-4 flex gap-2">
                    <Tooltip content="Export as ZIP" position="left">
                      <button
                        onClick={() => handleExport(project.result!.project_dir, 'zip')}
                        className="p-2 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
                        tabIndex={0}
                        aria-label={`Export project ${project.project_name || project.id} as ZIP`}
                        onKeyDown={(e) =>
                          handleKeyDown(e, () => handleExport(project.result!.project_dir, 'zip'))
                        }
                      >
                        <Download className="w-5 h-5" />
                      </button>
                    </Tooltip>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ProjectList

