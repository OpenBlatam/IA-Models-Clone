export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export function getUrlParams(url: string): Record<string, string> {
  try {
    const urlObj = new URL(url);
    const params: Record<string, string> = {};
    urlObj.searchParams.forEach((value, key) => {
      params[key] = value;
    });
    return params;
  } catch {
    return {};
  }
}

export function setUrlParam(url: string, key: string, value: string): string {
  try {
    const urlObj = new URL(url);
    urlObj.searchParams.set(key, value);
    return urlObj.toString();
  } catch {
    return url;
  }
}

export function removeUrlParam(url: string, key: string): string {
  try {
    const urlObj = new URL(url);
    urlObj.searchParams.delete(key);
    return urlObj.toString();
  } catch {
    return url;
  }
}

export function getDomain(url: string): string | null {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch {
    return null;
  }
}

export function getPath(url: string): string | null {
  try {
    const urlObj = new URL(url);
    return urlObj.pathname;
  } catch {
    return null;
  }
}

export function buildUrl(base: string, params: Record<string, string | number | boolean>): string {
  try {
    const urlObj = new URL(base);
    Object.entries(params).forEach(([key, value]) => {
      urlObj.searchParams.set(key, String(value));
    });
    return urlObj.toString();
  } catch {
    return base;
  }
}

