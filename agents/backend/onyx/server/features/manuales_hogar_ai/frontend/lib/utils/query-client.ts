import { QueryClient } from '@tanstack/react-query';
import { QUERY_CONFIG } from '../constants';

export const createQueryClient = (): QueryClient => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: QUERY_CONFIG.STALE_TIME,
        gcTime: QUERY_CONFIG.GC_TIME,
        refetchOnWindowFocus: false,
        retry: QUERY_CONFIG.RETRY,
        retryDelay: QUERY_CONFIG.RETRY_DELAY,
      },
      mutations: {
        retry: QUERY_CONFIG.RETRY,
        retryDelay: QUERY_CONFIG.RETRY_DELAY,
      },
    },
  });
};

