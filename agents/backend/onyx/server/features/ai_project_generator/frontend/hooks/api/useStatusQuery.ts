import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { GeneratorStatus, QueueResponse, Stats } from '@/types'

export const useStatusQuery = () => {
  return useQuery<GeneratorStatus>({
    queryKey: ['status'],
    queryFn: () => api.getStatus(),
    refetchInterval: 5000,
    staleTime: 3000,
  })
}

export const useQueueQuery = () => {
  return useQuery<QueueResponse>({
    queryKey: ['queue'],
    queryFn: () => api.getQueue(),
    refetchInterval: 5000,
    staleTime: 3000,
  })
}

export const useStatsQuery = () => {
  return useQuery<Stats>({
    queryKey: ['stats'],
    queryFn: () => api.getStats(),
    refetchInterval: 10000,
    staleTime: 5000,
  })
}

