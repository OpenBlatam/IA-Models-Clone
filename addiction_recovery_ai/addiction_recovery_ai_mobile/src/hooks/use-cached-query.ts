import { useQuery, UseQueryOptions, QueryFunction } from '@tanstack/react-query';
import { cacheManager } from '@/utils/cache-manager';

export function useCachedQuery<TData, TError = Error>(
  options: UseQueryOptions<TData, TError> & {
    cacheKey?: string;
    cacheTTL?: number;
    queryFn: QueryFunction<TData>;
  }
) {
  const { cacheKey, cacheTTL = 60000, queryFn, ...queryOptions } = options;

  const query = useQuery<TData, TError>({
    ...queryOptions,
    queryFn: async () => {
      if (cacheKey) {
        const cached = cacheManager.get<TData>(cacheKey);
        if (cached !== null) {
          return cached;
        }
      }

      const data = await queryFn();
      
      if (cacheKey) {
        cacheManager.set(cacheKey, data, cacheTTL);
      }

      return data;
    },
  });

  return query;
}
