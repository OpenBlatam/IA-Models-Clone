/**
 * URL utilities
 */

/**
 * Parse URL
 */
export const parseURL = (url: string): {
  protocol: string;
  host: string;
  path: string;
  query: Record<string, string>;
  hash: string;
} | null => {
  try {
    const urlObj = new URL(url);
    const query: Record<string, string> = {};
    urlObj.searchParams.forEach((value, key) => {
      query[key] = value;
    });

    return {
      protocol: urlObj.protocol,
      host: urlObj.host,
      path: urlObj.pathname,
      query,
      hash: urlObj.hash,
    };
  } catch {
    return null;
  }
};

/**
 * Build URL with query params
 */
export const buildURL = (
  base: string,
  params?: Record<string, string | number | boolean>
): string => {
  const url = new URL(base);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, String(value));
    });
  }
  return url.toString();
};

/**
 * Get query parameter from URL
 */
export const getQueryParam = (url: string, param: string): string | null => {
  try {
    const urlObj = new URL(url);
    return urlObj.searchParams.get(param);
  } catch {
    return null;
  }
};

/**
 * Remove query parameter from URL
 */
export const removeQueryParam = (url: string, param: string): string => {
  try {
    const urlObj = new URL(url);
    urlObj.searchParams.delete(param);
    return urlObj.toString();
  } catch {
    return url;
  }
};

/**
 * Encode URL
 */
export const encodeURL = (url: string): string => {
  return encodeURIComponent(url);
};

/**
 * Decode URL
 */
export const decodeURL = (url: string): string => {
  try {
    return decodeURIComponent(url);
  } catch {
    return url;
  }
};

/**
 * Check if URL is valid
 */
export const isValidURL = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

/**
 * Get domain from URL
 */
export const getDomain = (url: string): string | null => {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch {
    return null;
  }
};

/**
 * Get path from URL
 */
export const getPath = (url: string): string | null => {
  try {
    const urlObj = new URL(url);
    return urlObj.pathname;
  } catch {
    return null;
  }
};

