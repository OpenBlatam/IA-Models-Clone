/**
 * URL utility functions
 * Common URL operations
 */

/**
 * Parse query string to object
 */
export function parseQueryString(queryString: string): Record<string, string> {
  const params: Record<string, string> = {};
  
  if (!queryString) {
    return params;
  }

  const pairs = queryString.replace(/^\?/, '').split('&');
  
  for (const pair of pairs) {
    const [key, value] = pair.split('=');
    if (key) {
      params[decodeURIComponent(key)] = value ? decodeURIComponent(value) : '';
    }
  }

  return params;
}

/**
 * Build query string from object
 */
export function buildQueryString(params: Record<string, string | number | boolean>): string {
  const pairs: string[] = [];

  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined) {
      pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
    }
  }

  return pairs.length > 0 ? `?${pairs.join('&')}` : '';
}

/**
 * Get URL parameters
 */
export function getUrlParams(url: string): Record<string, string> {
  try {
    const urlObj = new URL(url);
    return parseQueryString(urlObj.search);
  } catch {
    return {};
  }
}

/**
 * Add parameters to URL
 */
export function addUrlParams(
  url: string,
  params: Record<string, string | number | boolean>
): string {
  try {
    const urlObj = new URL(url);
    const existingParams = parseQueryString(urlObj.search);
    const newParams = { ...existingParams, ...params };
    urlObj.search = buildQueryString(newParams);
    return urlObj.toString();
  } catch {
    return url;
  }
}

/**
 * Remove parameters from URL
 */
export function removeUrlParams(url: string, keys: string[]): string {
  try {
    const urlObj = new URL(url);
    const params = parseQueryString(urlObj.search);
    
    keys.forEach((key) => {
      delete params[key];
    });

    urlObj.search = buildQueryString(params);
    return urlObj.toString();
  } catch {
    return url;
  }
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get domain from URL
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
 * Get path from URL
 */
export function getPath(url: string): string | null {
  try {
    const urlObj = new URL(url);
    return urlObj.pathname;
  } catch {
    return null;
  }
}

