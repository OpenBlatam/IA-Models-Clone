/**
 * Custom hook for managing URL query parameters.
 * Provides reactive access to URL search parameters.
 */

import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { useCallback } from 'react';

/**
 * Options for useQueryParams hook.
 */
export interface UseQueryParamsOptions {
  replace?: boolean;
}

/**
 * Return type for useQueryParams hook.
 */
export interface UseQueryParamsReturn {
  params: URLSearchParams;
  getParam: (key: string, defaultValue?: string) => string | undefined;
  setParam: (key: string, value: string | number | null) => void;
  removeParam: (key: string) => void;
  clearParams: () => void;
  getAllParams: () => Record<string, string>;
}

/**
 * Custom hook for managing URL query parameters.
 * Provides reactive access and manipulation of URL search parameters.
 *
 * @param options - Hook options
 * @returns Query parameters state and handlers
 */
export function useQueryParams(
  options: UseQueryParamsOptions = {}
): UseQueryParamsReturn {
  const { replace = false } = options;
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  /**
   * Gets a query parameter value.
   */
  const getParam = useCallback(
    (key: string, defaultValue?: string): string | undefined => {
      return searchParams.get(key) || defaultValue;
    },
    [searchParams]
  );

  /**
   * Sets a query parameter value.
   */
  const setParam = useCallback(
    (key: string, value: string | number | null) => {
      const params = new URLSearchParams(searchParams.toString());

      if (value === null) {
        params.delete(key);
      } else {
        params.set(key, String(value));
      }

      const newUrl = `${pathname}?${params.toString()}`;
      
      if (replace) {
        router.replace(newUrl);
      } else {
        router.push(newUrl);
      }
    },
    [searchParams, pathname, router, replace]
  );

  /**
   * Removes a query parameter.
   */
  const removeParam = useCallback(
    (key: string) => {
      setParam(key, null);
    },
    [setParam]
  );

  /**
   * Clears all query parameters.
   */
  const clearParams = useCallback(() => {
    if (replace) {
      router.replace(pathname);
    } else {
      router.push(pathname);
    }
  }, [pathname, router, replace]);

  /**
   * Gets all query parameters as an object.
   */
  const getAllParams = useCallback((): Record<string, string> => {
    const params: Record<string, string> = {};
    searchParams.forEach((value, key) => {
      params[key] = value;
    });
    return params;
  }, [searchParams]);

  return {
    params: searchParams,
    getParam,
    setParam,
    removeParam,
    clearParams,
    getAllParams,
  };
}

