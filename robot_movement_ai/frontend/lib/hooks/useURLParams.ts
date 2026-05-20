import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { useCallback } from 'react';
import { parseQueryString, buildQueryString } from '@/lib/utils/url';

export function useURLParams() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const getParam = useCallback(
    (key: string): string | null => {
      return searchParams.get(key);
    },
    [searchParams]
  );

  const getParams = useCallback((): Record<string, string> => {
    return parseQueryString(searchParams.toString());
  }, [searchParams]);

  const setParam = useCallback(
    (key: string, value: string | number | boolean | null, replace: boolean = false) => {
      const params = new URLSearchParams(searchParams.toString());
      
      if (value === null || value === undefined) {
        params.delete(key);
      } else {
        params.set(key, String(value));
      }

      const queryString = params.toString();
      const url = queryString ? `${pathname}?${queryString}` : pathname;

      if (replace) {
        router.replace(url);
      } else {
        router.push(url);
      }
    },
    [searchParams, pathname, router]
  );

  const setParams = useCallback(
    (params: Record<string, string | number | boolean | null>, replace: boolean = false) => {
      const currentParams = new URLSearchParams(searchParams.toString());

      Object.entries(params).forEach(([key, value]) => {
        if (value === null || value === undefined) {
          currentParams.delete(key);
        } else {
          currentParams.set(key, String(value));
        }
      });

      const queryString = currentParams.toString();
      const url = queryString ? `${pathname}?${queryString}` : pathname;

      if (replace) {
        router.replace(url);
      } else {
        router.push(url);
      }
    },
    [searchParams, pathname, router]
  );

  const removeParam = useCallback(
    (key: string, replace: boolean = false) => {
      setParam(key, null, replace);
    },
    [setParam]
  );

  return {
    getParam,
    getParams,
    setParam,
    setParams,
    removeParam,
  };
}



