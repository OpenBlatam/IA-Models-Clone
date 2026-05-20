import { REFETCH_INTERVALS } from '@/lib/constants';

export const queryConfig = {
  defaultStaleTime: 5 * 60 * 1000, // 5 minutes
  defaultCacheTime: 10 * 60 * 1000, // 10 minutes
  refetchOnWindowFocus: true,
  retry: 3,
  refetchIntervals: REFETCH_INTERVALS,
} as const;



