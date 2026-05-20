export const buildQueryString = (params: Record<string, string | number | boolean | null | undefined>): string => {
  const searchParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      searchParams.append(key, String(value));
    }
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

export const getQueryParam = (key: string, defaultValue?: string): string | null => {
  if (typeof window === 'undefined') return defaultValue ?? null;
  
  const params = new URLSearchParams(window.location.search);
  return params.get(key) ?? defaultValue ?? null;
};

export const getAllQueryParams = (): Record<string, string> => {
  if (typeof window === 'undefined') return {};
  
  const params: Record<string, string> = {};
  const searchParams = new URLSearchParams(window.location.search);
  
  searchParams.forEach((value, key) => {
    params[key] = value;
  });
  
  return params;
};

export const setQueryParam = (key: string, value: string | number | boolean | null, replace = false): void => {
  if (typeof window === 'undefined') return;
  
  const url = new URL(window.location.href);
  
  if (value === null) {
    url.searchParams.delete(key);
  } else {
    url.searchParams.set(key, String(value));
  }
  
  if (replace) {
    window.history.replaceState({}, '', url.toString());
  } else {
    window.history.pushState({}, '', url.toString());
  }
};

export const removeQueryParam = (key: string, replace = false): void => {
  setQueryParam(key, null, replace);
};

export const updateQueryParams = (params: Record<string, string | number | boolean | null>, replace = false): void => {
  if (typeof window === 'undefined') return;
  
  const url = new URL(window.location.href);
  
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      url.searchParams.delete(key);
    } else {
      url.searchParams.set(key, String(value));
    }
  });
  
  if (replace) {
    window.history.replaceState({}, '', url.toString());
  } else {
    window.history.pushState({}, '', url.toString());
  }
};

