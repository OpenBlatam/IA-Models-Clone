import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { ProjectRequest } from '@/types'
import toast from 'react-hot-toast'

export const useGenerateProjectMutation = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: ProjectRequest) => api.generate(request),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['queue'] })
      queryClient.invalidateQueries({ queryKey: ['status'] })
      toast.success('Project added to queue successfully!')
      return response.project_id
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to generate project')
    },
  })
}

export const useStartGeneratorMutation = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.startGenerator(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] })
      toast.success('Generator started successfully!')
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to start generator')
    },
  })
}

export const useStopGeneratorMutation = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.stopGenerator(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['status'] })
      toast.success('Generator stopped successfully!')
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to stop generator')
    },
  })
}

