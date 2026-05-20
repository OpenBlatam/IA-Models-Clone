export const parseQueryString = (search?: string): Record<string, string> => {
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

export const buildQueryString = (params: Record<string, string | number | boolean | null | undefined>): string => {
  const urlParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      urlParams.append(key, String(value));
    }
  });

  return urlParams.toString();
};

export const updateQueryParam = (key: string, value: string | null, replace = false): void => {
  if (typeof window === 'undefined') {
    return;
  }

  const url = new URL(window.location.href);
  if (value === null) {
    url.searchParams.delete(key);
  } else {
    url.searchParams.set(key, value);
  }

  if (replace) {
    window.history.replaceState({}, '', url.toString());
  } else {
    window.history.pushState({}, '', url.toString());
  }
};

export const getQueryParam = (key: string, defaultValue?: string): string | null => {
  if (typeof window === 'undefined') {
    return defaultValue || null;
  }

  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(key) || defaultValue || null;
};

export const removeQueryParam = (key: string, replace = false): void => {
  updateQueryParam(key, null, replace);
};



