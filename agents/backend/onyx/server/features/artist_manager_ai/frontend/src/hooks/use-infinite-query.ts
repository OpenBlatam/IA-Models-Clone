import { useInfiniteQuery, InfiniteData } from '@tanstack/react-query';

interface UseInfiniteQueryOptions<TData, TError = Error> {
  queryKey: readonly unknown[];
  queryFn: ({ pageParam }: { pageParam?: number }) => Promise<TData>;
  getNextPageParam: (lastPage: TData, allPages: TData[]) => number | undefined;
  initialPageParam?: number;
  enabled?: boolean;
}

export const useInfiniteQueryCustom = <TData, TError = Error>({
  queryKey,
  queryFn,
  getNextPageParam,
  initialPageParam = 0,
  enabled = true,
}: UseInfiniteQueryOptions<TData, TError>) => {
  return useInfiniteQuery<TData, TError, InfiniteData<TData>, readonly unknown[], number>({
    queryKey,
    queryFn,
    getNextPageParam,
    initialPageParam,
    enabled,
  });
};

