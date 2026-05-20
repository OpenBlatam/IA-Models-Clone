/**
 * Custom hook for monitoring API health status.
 * Provides real-time connection status and automatic health checks.
 */

import { useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { checkApiHealth } from '@/lib/api/client';

export interface ApiHealthStatus {
  isHealthy: boolean;
  isLoading: boolean;
  lastChecked: number | null;
  error: Error | null;
  message: string;
}

/**
 * Hook for monitoring API health.
 * @param options - Configuration options
 * @param options.enabled - Whether to enable health checks (default: true)
 * @param options.refetchInterval - Interval for automatic health checks in ms (default: 30000)
 * @returns API health status and manual refresh function
 */
export function useApiHealth(options: {
  enabled?: boolean;
  refetchInterval?: number;
} = {}) {
  const { enabled = true, refetchInterval = 30000 } = options;

  const {
    data: healthData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['api', 'health', 'music'], // Specific key for health checks
    queryFn: async () => {
      const result = await checkApiHealth();
      return {
        isHealthy: result.status === 'healthy',
        message: result.message,
        timestamp: result.timestamp,
      };
    },
    enabled,
    refetchInterval: enabled ? refetchInterval : false,
    retry: 2,
    retryDelay: 1000,
    staleTime: 5000, // Consider stale after 5 seconds
  });

  /**
   * Manually refresh health check.
   */
  const refreshHealth = useCallback(() => {
    refetch();
  }, [refetch]);

  return {
    isHealthy: healthData?.isHealthy ?? false,
    isLoading,
    lastChecked: healthData?.timestamp ?? null,
    error: error as Error | null,
    message: healthData?.message ?? 'Checking API status...',
    refreshHealth,
  };
}

