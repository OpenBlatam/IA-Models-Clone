/**
 * Custom hook for cookie management.
 * Provides reactive cookie access and manipulation.
 */

import { useState, useCallback, useEffect } from 'react';
import { getCookie, setCookie, removeCookie } from '../utils/cookie';

/**
 * Options for useCookie hook.
 */
export interface UseCookieOptions {
  defaultValue?: string;
  expires?: Date | number;
  path?: string;
  domain?: string;
  secure?: boolean;
  sameSite?: 'strict' | 'lax' | 'none';
}

/**
 * Return type for useCookie hook.
 */
export interface UseCookieReturn {
  value: string | null;
  setValue: (value: string, options?: UseCookieOptions) => void;
  removeValue: () => void;
}

/**
 * Custom hook for cookie management.
 * Provides reactive cookie access with automatic updates.
 *
 * @param name - Cookie name
 * @param options - Cookie options
 * @returns Cookie value and handlers
 */
export function useCookie(
  name: string,
  options: UseCookieOptions = {}
): UseCookieReturn {
  const [value, setValueState] = useState<string | null>(() => {
    if (typeof window === 'undefined') {
      return options.defaultValue || null;
    }
    return getCookie(name) || options.defaultValue || null;
  });

  const setValue = useCallback(
    (newValue: string, cookieOptions?: UseCookieOptions) => {
      const mergedOptions = { ...options, ...cookieOptions };
      setCookie(name, newValue, mergedOptions);
      setValueState(newValue);
    },
    [name, options]
  );

  const removeValue = useCallback(() => {
    removeCookie(name, {
      path: options.path,
      domain: options.domain,
    });
    setValueState(null);
  }, [name, options.path, options.domain]);

  // Listen for cookie changes (if needed)
  useEffect(() => {
    const checkCookie = () => {
      const currentValue = getCookie(name);
      if (currentValue !== value) {
        setValueState(currentValue);
      }
    };

    // Check periodically (every second)
    const interval = setInterval(checkCookie, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [name, value]);

  return {
    value,
    setValue,
    removeValue,
  };
}

