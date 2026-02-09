import { useLocalSearchParams } from 'expo-router';
import { useMemo } from 'react';

/**
 * Hook to get and parse query parameters with type safety
 */
export function useQueryParams<T extends Record<string, string>>(): Partial<T> {
  const params = useLocalSearchParams();

  return useMemo(() => {
    const typedParams: Partial<T> = {};

    for (const [key, value] of Object.entries(params)) {
      if (typeof value === 'string') {
        typedParams[key as keyof T] = value as T[keyof T];
      }
    }

    return typedParams;
  }, [params]);
}

/**
 * Hook to get a single query parameter
 */
export function useQueryParam<T extends string = string>(
  key: string,
  defaultValue?: T
): T | undefined {
  const params = useQueryParams<Record<string, T>>();
  return (params[key] as T | undefined) ?? defaultValue;
}

