export const getQueryParams = (search?: string): Record<string, string> => {
  const params: Record<string, string> = {};
  const queryString = search || (typeof window !== 'undefined' ? window.location.search : '');

  if (!queryString) {
    return params;
  }

  const urlParams = new URLSearchParams(queryString);
  urlParams.forEach((value, key) => {
    params[key] = value;
  });

  return params;
};

export const setQueryParams = (params: Record<string, string | null>, replace = false): void => {
  if (typeof window === 'undefined') {
    return;
  }

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

export const removeQueryParams = (keys: string[], replace = false): void => {
  if (typeof window === 'undefined') {
    return;
  }

  const url = new URL(window.location.href);
  keys.forEach((key) => {
    url.searchParams.delete(key);
  });

  if (replace) {
    window.history.replaceState({}, '', url.toString());
  } else {
    window.history.pushState({}, '', url.toString());
  }
};

export const buildUrl = (base: string, params?: Record<string, string | number | boolean>): string => {
  const url = new URL(base);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, String(value));
    });
  }
  return url.toString();
};

export const parseUrl = (url: string): {
  protocol: string;
  host: string;
  hostname: string;
  port: string;
  pathname: string;
  search: string;
  hash: string;
} => {
  const parsed = new URL(url);
  return {
    protocol: parsed.protocol,
    host: parsed.host,
    hostname: parsed.hostname,
    port: parsed.port,
    pathname: parsed.pathname,
    search: parsed.search,
    hash: parsed.hash,
  };
};



