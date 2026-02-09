import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { Project } from '@/types'
import toast from 'react-hot-toast'

export const useProjectsQuery = (filters?: { status?: string }) => {
  return useQuery({
    queryKey: ['projects', filters],
    queryFn: async () => {
      const response = await api.getProjects(filters || {})
      return response.projects
    },
    staleTime: 30000,
    refetchInterval: 5000,
  })
}

export const useProjectQuery = (projectId: string) => {
  return useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.getProject(projectId),
    enabled: !!projectId,
  })
}

export const useDeleteProjectMutation = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (projectId: string) => api.deleteProject(projectId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      toast.success('Project deleted successfully')
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to delete project')
    },
  })
}

export const useExportProjectMutation = () => {
  return useMutation({
    mutationFn: ({ projectPath, format }: { projectPath: string; format: 'zip' | 'tar' }) => {
      return format === 'zip' ? api.exportZip(projectPath) : api.exportTar(projectPath)
    },
    onSuccess: (response) => {
      const exportPath = response.zip_path || response.tar_path
      toast.success(`Project exported to: ${exportPath}`)
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to export project')
    },
  })
}

