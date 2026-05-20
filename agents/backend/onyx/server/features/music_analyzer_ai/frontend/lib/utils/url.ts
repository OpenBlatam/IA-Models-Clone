/**
 * URL utility functions.
 * Provides helper functions for URL manipulation and validation.
 */

/**
 * Parses query parameters from a URL string.
 * @param url - URL string or search string
 * @returns Object with query parameters
 */
export function parseQueryParams(
  url: string | URLSearchParams
): Record<string, string> {
  const params: Record<string, string> = {};

  if (typeof url === 'string') {
    try {
      const urlObj = new URL(url, typeof window !== 'undefined' ? window.location.origin : 'http://localhost');
      urlObj.searchParams.forEach((value, key) => {
        params[key] = value;
      });
    } catch {
      // If URL parsing fails, try parsing as search string
      const searchParams = new URLSearchParams(url);
      searchParams.forEach((value, key) => {
        params[key] = value;
      });
    }
  } else {
    url.forEach((value, key) => {
      params[key] = value;
    });
  }

  return params;
}

/**
 * Builds a URL with query parameters.
 * @param baseUrl - Base URL
 * @param params - Query parameters object
 * @returns URL string with query parameters
 */
export function buildUrl(
  baseUrl: string,
  params: Record<string, string | number | boolean | null | undefined>
): string {
  const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
  const url = new URL(baseUrl, origin);
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  });

  return url.toString();
}

/**
 * Gets a query parameter from the current URL.
 * @param key - Parameter key
 * @param defaultValue - Default value if not found
 * @returns Parameter value or default
 */
export function getQueryParam(
  key: string,
  defaultValue?: string
): string | undefined {
  if (typeof window === 'undefined') {
    return defaultValue;
  }

  const params = new URLSearchParams(window.location.search);
  return params.get(key) || defaultValue;
}

/**
 * Sets a query parameter in the current URL.
 * @param key - Parameter key
 * @param value - Parameter value
 * @param replace - Whether to replace current history entry
 */
export function setQueryParam(
  key: string,
  value: string | number | null,
  replace: boolean = false
): void {
  if (typeof window === 'undefined') {
    return;
  }

  const url = new URL(window.location.href);
  
  if (value === null) {
    url.searchParams.delete(key);
  } else {
    url.searchParams.set(key, String(value));
  }

  if (replace) {
    window.history.replaceState({}, '', url);
  } else {
    window.history.pushState({}, '', url);
  }
}

/**
 * Removes a query parameter from the current URL.
 * @param key - Parameter key to remove
 * @param replace - Whether to replace current history entry
 */
export function removeQueryParam(key: string, replace: boolean = false): void {
  setQueryParam(key, null, replace);
}

/**
 * Gets the base URL without query parameters.
 * @param url - URL string
 * @returns Base URL without query string
 */
export function getBaseUrl(url?: string): string {
  const defaultUrl = typeof window !== 'undefined' ? window.location.href : 'http://localhost';
  const targetUrl = url || defaultUrl;
  
  try {
    const urlObj = new URL(targetUrl);
    return `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
  } catch {
    return targetUrl.split('?')[0];
  }
}

/**
 * Checks if a URL is absolute.
 * @param url - URL to check
 * @returns True if URL is absolute
 */
export function isAbsoluteUrl(url: string): boolean {
  if (typeof url !== 'string') {
    return false;
  }

  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Converts a relative URL to absolute.
 * @param url - Relative URL
 * @param baseUrl - Base URL (default: current location)
 * @returns Absolute URL
 */
export function toAbsoluteUrl(
  url: string,
  baseUrl: string = typeof window !== 'undefined' ? window.location.origin : ''
): string {
  if (isAbsoluteUrl(url)) {
    return url;
  }

  try {
    return new URL(url, baseUrl).toString();
  } catch {
    return url;
  }
}

/**
 * Gets the domain from a URL.
 * @param url - URL string
 * @returns Domain name
 */
export function getDomain(url: string): string | null {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch {
    return null;
  }
}

/**
 * Gets the path from a URL.
 * @param url - URL string
 * @returns Path name
 */
export function getPath(url: string): string | null {
  try {
    const urlObj = new URL(url);
    return urlObj.pathname;
  } catch {
    const match = url.match(/^[^?#]*/);
    return match ? match[0] : null;
  }
}

