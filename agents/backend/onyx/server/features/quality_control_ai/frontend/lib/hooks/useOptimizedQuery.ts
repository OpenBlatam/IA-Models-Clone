import { useQuery, UseQueryOptions } from '@tanstack/react-query';
import { useMemo } from 'react';

export const useOptimizedQuery = <TData, TError = Error>(
  options: UseQueryOptions<TData, TError>
) => {
  const optimizedOptions = useMemo(
    () => ({
      ...options,
      select: options.select,
      structuralSharing: true,
      refetchOnMount: false,
      refetchOnWindowFocus: false,
    }),
    [options]
  );

  return useQuery(optimizedOptions);
};

