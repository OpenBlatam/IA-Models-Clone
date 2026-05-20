/**
 * React Query Client Configuration
 * Centralized configuration for React Query with optimized defaults
 */

import { QueryClient } from '@tanstack/react-query';
import { DEFAULTS } from '@/lib/config/constants';
import { getErrorMessage } from '@/lib/errors/handler';
import { isNetworkError } from '@/lib/errors/handler';

/**
 * Creates a new QueryClient with optimized configuration
 * @returns Configured QueryClient instance
 */
export const createQueryClient = (): QueryClient => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: (failureCount, error) => {
          // Don't retry on network errors or 4xx errors
          if (isNetworkError(error) || failureCount >= DEFAULTS.QUERY_RETRY) {
            return false;
          }
          return true;
        },
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
        staleTime: DEFAULTS.QUERY_STALE_TIME,
        gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      },
      mutations: {
        retry: (failureCount, error) => {
          // Don't retry mutations on network errors
          if (isNetworkError(error) || failureCount >= DEFAULTS.QUERY_RETRY) {
            return false;
          }
          return true;
        },
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
        onError: (error) => {
          // Global error handler for mutations
          console.error('Mutation error:', getErrorMessage(error));
        },
      },
    },
  });
};

/**
 * Default QueryClient instance
 * Use this for the QueryClientProvider
 */
export const queryClient = createQueryClient();


