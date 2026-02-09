import { useQuery, UseQueryOptions } from 'react-query';
import { AxiosError } from 'axios';

interface UseQueryWithErrorOptions<TData, TError = AxiosError> extends Omit<UseQueryOptions<TData, TError>, 'onError'> {
  onError?: (error: TError) => void;
}

export const useQueryWithError = <TData, TError = AxiosError>(
  options: UseQueryWithErrorOptions<TData, TError>
) => {
  const { onError, ...queryOptions } = options;

  return useQuery<TData, TError>({
    ...queryOptions,
    onError: (error) => {
      if (onError) {
        onError(error);
      }
    },
  });
};



