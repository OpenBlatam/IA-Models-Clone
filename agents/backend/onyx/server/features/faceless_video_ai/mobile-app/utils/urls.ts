/**
 * URL and deep linking utilities
 */

import * as Linking from 'expo-linking';

export function buildUrl(baseUrl: string, params?: Record<string, string | number | boolean>): string {
  const url = new URL(baseUrl);
  
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, String(value));
    });
  }
  
  return url.toString();
}

export function parseUrl(url: string): {
  protocol: string;
  host: string;
  path: string;
  params: Record<string, string>;
} {
  try {
    const parsed = new URL(url);
    const params: Record<string, string> = {};
    
    parsed.searchParams.forEach((value, key) => {
      params[key] = value;
    });
    
    return {
      protocol: parsed.protocol,
      host: parsed.host,
      path: parsed.pathname,
      params,
    };
  } catch {
    return {
      protocol: '',
      host: '',
      path: url,
      params: {},
    };
  }
}

export function getQueryParams(url: string): Record<string, string> {
  const parsed = parseUrl(url);
  return parsed.params;
}

export function addQueryParams(url: string, params: Record<string, string | number | boolean>): string {
  const parsed = new URL(url);
  
  Object.entries(params).forEach(([key, value]) => {
    parsed.searchParams.set(key, String(value));
  });
  
  return parsed.toString();
}

export function removeQueryParams(url: string, keys: string[]): string {
  const parsed = new URL(url);
  
  keys.forEach((key) => {
    parsed.searchParams.delete(key);
  });
  
  return parsed.toString();
}

export async function canOpenUrl(url: string): Promise<boolean> {
  try {
    return await Linking.canOpenURL(url);
  } catch {
    return false;
  }
}

export async function openUrl(url: string): Promise<boolean> {
  try {
    const canOpen = await canOpenUrl(url);
    if (canOpen) {
      await Linking.openURL(url);
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

export function getDeepLink(path: string, params?: Record<string, string | number | boolean>): string {
  const scheme = Linking.createURL('/');
  const baseUrl = scheme.replace(/\/$/, '');
  const url = `${baseUrl}${path.startsWith('/') ? path : `/${path}`}`;
  
  if (params && Object.keys(params).length > 0) {
    return addQueryParams(url, params);
  }
  
  return url;
}

export function parseDeepLink(url: string): {
  scheme: string;
  host: string;
  path: string;
  params: Record<string, string>;
} {
  try {
    const parsed = Linking.parse(url);
    
    return {
      scheme: parsed.scheme || '',
      host: parsed.hostname || '',
      path: parsed.path || '',
      params: (parsed.queryParams || {}) as Record<string, string>,
    };
  } catch {
    return {
      scheme: '',
      host: '',
      path: url,
      params: {},
    };
  }
}

export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export function isAbsoluteUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

export function normalizeUrl(url: string): string {
  if (isAbsoluteUrl(url)) {
    return url;
  }
  
  if (url.startsWith('//')) {
    return `https:${url}`;
  }
  
  if (url.startsWith('/')) {
    return url;
  }
  
  return `/${url}`;
}

export function getDomain(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.hostname;
  } catch {
    return '';
  }
}

export function getProtocol(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.protocol.replace(':', '');
  } catch {
    return '';
  }
}

export function getPath(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.pathname;
  } catch {
    return url;
  }
}

export function encodeUrl(url: string): string {
  try {
    return encodeURIComponent(url);
  } catch {
    return url;
  }
}

export function decodeUrl(encodedUrl: string): string {
  try {
    return decodeURIComponent(encodedUrl);
  } catch {
    return encodedUrl;
  }
}

export function sanitizeUrl(url: string): string {
  // Remove potentially dangerous characters
  return url
    .replace(/[<>"']/g, '')
    .trim();
}


