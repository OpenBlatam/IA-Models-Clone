import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api'
import type {
  CodeImproveRequest,
  CodeImproveResponse,
} from '@/lib/api/types'
import toast from 'react-hot-toast'

export const useImproveCode = () => {
  return useMutation<CodeImproveResponse, Error, CodeImproveRequest>({
    mutationFn: (request: CodeImproveRequest) =>
      apiClient.improveCode(request),
    onSuccess: (data) => {
      toast.success(
        `Code improved! ${data.improvements_applied} improvements applied.`
      )
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to improve code')
    },
  })
}

export const useImproveCodeText = () => {
  return useMutation<
    CodeImproveResponse,
    Error,
    { code: string; context?: string; modelId?: string }
  >({
    mutationFn: ({ code, context, modelId }) =>
      apiClient.improveCodeText(code, context, modelId),
    onSuccess: (data) => {
      toast.success(
        `Code improved! ${data.improvements_applied} improvements applied.`
      )
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Failed to improve code')
    },
  })
}




