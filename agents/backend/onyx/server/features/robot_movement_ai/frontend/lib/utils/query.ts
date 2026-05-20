/**
 * Query string utilities
 */

import { parseQueryString, buildQueryString } from './url';

// Get query param
export function getQueryParam(key: string, defaultValue: string = ''): string {
  if (typeof window === 'undefined') {
    return defaultValue;
  }

  const params = parseQueryString(window.location.search);
  return params[key] || defaultValue;
}

// Set query param
export function setQueryParam(key: string, value: string | number | boolean | null) {
  if (typeof window === 'undefined') {
    return;
  }

  const params = parseQueryString(window.location.search);
  
  if (value === null || value === undefined) {
    delete params[key];
  } else {
    params[key] = String(value);
  }

  const queryString = buildQueryString(params);
  const newUrl = queryString
    ? `${window.location.pathname}?${queryString}`
    : window.location.pathname;

  window.history.pushState({}, '', newUrl);
}

// Remove query param
export function removeQueryParam(key: string) {
  setQueryParam(key, null);
}

// Get all query params
export function getAllQueryParams(): Record<string, string> {
  if (typeof window === 'undefined') {
    return {};
  }

  return parseQueryString(window.location.search);
}

// Clear all query params
export function clearQueryParams() {
  if (typeof window === 'undefined') {
    return;
  }

  window.history.pushState({}, '', window.location.pathname);
}



