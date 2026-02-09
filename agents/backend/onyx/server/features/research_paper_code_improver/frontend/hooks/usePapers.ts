import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import toast from 'react-hot-toast'

export const usePapers = (limit = 100) => {
  return useQuery({
    queryKey: ['papers', limit],
    queryFn: () => apiClient.listPapers(limit),
  })
}

export const usePaper = (paperId: string | null) => {
  return useQuery({
    queryKey: ['paper', paperId],
    queryFn: () => {
      if (!paperId) throw new Error('Paper ID is required')
      return apiClient.getPaper(paperId)
    },
    enabled: !!paperId,
  })
}

export const useUploadPaper = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (file: File) => apiClient.uploadPaper(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
      toast.success('Paper uploaded successfully!')
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to upload paper')
    },
  })
}

export const useProcessLink = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (url: string) =>
      apiClient.processLink({ url, download: true }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['papers'] })
      toast.success('Paper processed successfully!')
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to process link')
    },
  })
}




