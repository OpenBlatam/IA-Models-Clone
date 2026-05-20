export const getQueryParam = (key: string, url?: string): string | null => {
  const searchParams = new URLSearchParams(url ? new URL(url).search : window.location.search);
  return searchParams.get(key);
};

export const getAllQueryParams = (url?: string): Record<string, string> => {
  const searchParams = new URLSearchParams(url ? new URL(url).search : window.location.search);
  const params: Record<string, string> = {};
  searchParams.forEach((value, key) => {
    params[key] = value;
  });
  return params;
};

export const setQueryParam = (key: string, value: string, url?: string): string => {
  const urlObj = url ? new URL(url) : new URL(window.location.href);
  urlObj.searchParams.set(key, value);
  return urlObj.toString();
};

export const removeQueryParam = (key: string, url?: string): string => {
  const urlObj = url ? new URL(url) : new URL(window.location.href);
  urlObj.searchParams.delete(key);
  return urlObj.toString();
};

export const buildQueryString = (params: Record<string, string | number | boolean>): string => {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    searchParams.append(key, String(value));
  });
  return searchParams.toString();
};

export const parseQueryString = (queryString: string): Record<string, string> => {
  const params: Record<string, string> = {};
  const searchParams = new URLSearchParams(queryString);
  searchParams.forEach((value, key) => {
    params[key] = value;
  });
  return params;
};

export const updateUrl = (params: Record<string, string | null>, replace = false): void => {
  const url = new URL(window.location.href);
  Object.entries(params).forEach(([key, value]) => {
    if (value === null) {
      url.searchParams.delete(key);
    } else {
      url.searchParams.set(key, value);
    }
  });

  if (replace) {
    window.history.replaceState({}, '', url.toString());
  } else {
    window.history.pushState({}, '', url.toString());
  }
};

export const getHash = (): string => {
  return window.location.hash.slice(1);
};

export const setHash = (hash: string, replace = false): void => {
  if (replace) {
    window.history.replaceState(null, '', `#${hash}`);
  } else {
    window.history.pushState(null, '', `#${hash}`);
  }
};

export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

export const getDomain = (url: string): string => {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch {
    return '';
  }
};

export const getPathname = (url?: string): string => {
  if (url) {
    try {
      return new URL(url).pathname;
    } catch {
      return '';
    }
  }
  return window.location.pathname;
};

