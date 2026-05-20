'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import { getErrorMessage } from '@/lib/utils'
import { FILTER_TYPES, type FilterType, REFRESH_INTERVALS } from '@/lib/constants'
import type { Project } from '@/types'
import { Folder } from 'lucide-react'
import { ErrorMessage, LoadingSpinner, EmptyState, Button, Card, SearchInput, Pagination } from '@/components/ui'
import { useSearch, usePagination } from '@/hooks/ui'
import ProjectCard from './ProjectCard'

const ProjectList = () => {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterType>(FILTER_TYPES.ALL)

  const { searchQuery, setSearchQuery, filteredData: filteredProjects, clearSearch } = useSearch<Project>({
    data: projects,
    searchKeys: ['project_name', 'description', 'author', 'status'],
  })

  const { currentPage, totalPages, paginatedData, goToPage } = usePagination<Project>({
    data: filteredProjects,
    itemsPerPage: 12,
  })

  const loadProjects = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const params: { status?: string } = {}
      if (filter !== FILTER_TYPES.ALL) {
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
    const interval = setInterval(loadProjects, REFRESH_INTERVALS.PROJECTS)
    return () => clearInterval(interval)
  }, [loadProjects])

  const handleExport = useCallback(async (projectPath: string) => {
    try {
      const response = await api.exportZip(projectPath)
      alert(`Project exported to: ${response.zip_path}`)
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      alert(errorMessage || 'Failed to export project')
      console.error('Error exporting project:', err)
    }
  }, [])

  const handleFilterChange = useCallback((newFilter: FilterType) => {
    setFilter(newFilter)
    setError(null)
    goToPage(1)
  }, [goToPage])

  if (loading && projects.length === 0) {
    return (
      <Card>
        <LoadingSpinner size="lg" text="Loading projects..." />
      </Card>
    )
  }

  const filterOptions = [
    { value: FILTER_TYPES.ALL, label: 'All' },
    { value: FILTER_TYPES.COMPLETED, label: 'Completed' },
    { value: FILTER_TYPES.FAILED, label: 'Failed' },
    { value: FILTER_TYPES.PROCESSING, label: 'Processing' },
  ] as const

  return (
    <Card>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Projects</h2>
        <div className="flex flex-col sm:flex-row gap-3">
          <SearchInput
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Search projects..."
            onClear={clearSearch}
            className="w-full sm:w-64"
          />
          <div className="flex gap-2" role="group" aria-label="Filter projects">
            {filterOptions.map((option) => (
              <Button
                key={option.value}
                variant={filter === option.value ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => handleFilterChange(option.value)}
                aria-label={`Filter by ${option.label}`}
                aria-pressed={filter === option.value}
              >
                {option.label}
              </Button>
            ))}
          </div>
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
            filter === FILTER_TYPES.ALL
              ? 'No projects have been generated yet. Create your first project to get started!'
              : `No ${filter} projects found. Try a different filter.`
          }
        />
      ) : filteredProjects.length === 0 ? (
        <EmptyState
          icon={Folder}
          title="No matching projects"
          description="Try adjusting your search or filter criteria."
        />
      ) : (
        <>
          <div className="mb-4 text-sm text-gray-600">
            Showing {paginatedData.length} of {filteredProjects.length} projects
            {searchQuery && ` matching "${searchQuery}"`}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {paginatedData.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onExport={project.result ? () => handleExport(project.result!.project_dir) : undefined}
              />
            ))}
          </div>
          {totalPages > 1 && (
            <Pagination
              currentPage={currentPage}
              totalPages={totalPages}
              onPageChange={goToPage}
            />
          )}
        </>
      )}
    </Card>
  )
}

export default ProjectList

