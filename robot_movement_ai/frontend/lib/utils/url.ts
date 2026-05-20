/**
 * URL utilities
 */

// Parse query string
export function parseQueryString(queryString: string): Record<string, string> {
  const params: Record<string, string> = {};
  const searchParams = new URLSearchParams(queryString);

  for (const [key, value] of searchParams.entries()) {
    params[key] = value;
  }

  return params;
}

// Build query string
export function buildQueryString(params: Record<string, string | number | boolean | null | undefined>): string {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      searchParams.append(key, String(value));
    }
  });

  return searchParams.toString();
}

// Update URL without reload
export function updateURL(params: Record<string, string | number | boolean | null | undefined>, replace: boolean = false) {
  if (typeof window === 'undefined') {
    return;
  }

  const url = new URL(window.location.href);
  const queryString = buildQueryString(params);

  if (replace) {
    window.history.replaceState({}, '', `${url.pathname}?${queryString}`);
  } else {
    window.history.pushState({}, '', `${url.pathname}?${queryString}`);
  }
}

// Get current URL params
export function getURLParams(): Record<string, string> {
  if (typeof window === 'undefined') {
    return {};
  }

  return parseQueryString(window.location.search);
}

// Check if URL is external
export function isExternalURL(url: string): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  try {
    const urlObj = new URL(url, window.location.href);
    return urlObj.origin !== window.location.origin;
  } catch {
    return false;
  }
}



