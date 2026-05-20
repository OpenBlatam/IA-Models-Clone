import { QueryClient } from '@tanstack/react-query';

export function createOptimizedQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5 * 60 * 1000, // 5 minutes
        gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
        refetchOnWindowFocus: false,
        refetchOnMount: false,
        refetchOnReconnect: false,
        retry: 1,
        retryDelay: 1000,
      },
      mutations: {
        retry: 0,
      },
    },
  });
}

