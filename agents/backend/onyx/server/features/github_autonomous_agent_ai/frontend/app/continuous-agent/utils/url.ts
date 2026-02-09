export const parseUrl = (url: string): URL => {
  try {
    return new URL(url);
  } catch {
    return new URL(url, window.location.origin);
  }
};

export const buildUrl = (
  base: string,
  params?: Record<string, string | number | boolean | null | undefined>
): string => {
  const url = new URL(base, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url.toString();
};

export const parseQueryString = (
  queryString: string
): Record<string, string> => {
  const params = new URLSearchParams(queryString);
  const result: Record<string, string> = {};
  params.forEach((value, key) => {
    result[key] = value;
  });
  return result;
};

export const buildQueryString = (
  params: Record<string, string | number | boolean | null | undefined>
): string => {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      searchParams.set(key, String(value));
    }
  });
  return searchParams.toString();
};

export const getQueryParam = (url: string, key: string): string | null => {
  const urlObj = parseUrl(url);
  return urlObj.searchParams.get(key);
};

export const setQueryParam = (
  url: string,
  key: string,
  value: string | number | boolean | null
): string => {
  const urlObj = parseUrl(url);
  if (value === null) {
    urlObj.searchParams.delete(key);
  } else {
    urlObj.searchParams.set(key, String(value));
  }
  return urlObj.toString();
};

export const removeQueryParam = (url: string, key: string): string => {
  const urlObj = parseUrl(url);
  urlObj.searchParams.delete(key);
  return urlObj.toString();
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
  const urlObj = parseUrl(url);
  return urlObj.hostname;
};

export const getPath = (url: string): string => {
  const urlObj = parseUrl(url);
  return urlObj.pathname;
};

export const getProtocol = (url: string): string => {
  const urlObj = parseUrl(url);
  return urlObj.protocol.replace(":", "");
};

export const isAbsoluteUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

export const isRelativeUrl = (url: string): boolean => {
  return !isAbsoluteUrl(url);
};

export const joinPaths = (...paths: string[]): string => {
  return paths
    .filter(Boolean)
    .join("/")
    .replace(/\/+/g, "/")
    .replace(/\/$/, "");
};

export const normalizeUrl = (url: string): string => {
  try {
    const urlObj = new URL(url);
    urlObj.pathname = urlObj.pathname.replace(/\/+/g, "/");
    return urlObj.toString();
  } catch {
    return url;
  }
};





