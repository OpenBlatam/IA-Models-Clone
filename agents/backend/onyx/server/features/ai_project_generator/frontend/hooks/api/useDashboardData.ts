import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import { getErrorMessage } from '@/lib/utils'
import { REFRESH_INTERVALS } from '@/lib/constants'
import type { GeneratorStatus, QueueResponse, Stats } from '@/types'

interface UseDashboardDataReturn {
  status: GeneratorStatus | null
  queue: QueueResponse | null
  stats: Stats | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

export const useDashboardData = (autoRefresh = true): UseDashboardDataReturn => {
  const [status, setStatus] = useState<GeneratorStatus | null>(null)
  const [queue, setQueue] = useState<QueueResponse | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const [statusData, queueData, statsData] = await Promise.all([
        api.getStatus(),
        api.getQueue(),
        api.getStats(),
      ])
      setStatus(statusData)
      setQueue(queueData)
      setStats(statsData)
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage)
      console.error('Error loading dashboard data:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()

    if (autoRefresh) {
      const interval = setInterval(loadData, REFRESH_INTERVALS.DASHBOARD)
      return () => clearInterval(interval)
    }
  }, [loadData, autoRefresh])

  return {
    status,
    queue,
    stats,
    loading,
    error,
    refresh: loadData,
  }
}

