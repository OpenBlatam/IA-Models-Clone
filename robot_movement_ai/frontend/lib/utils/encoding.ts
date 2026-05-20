/**
 * Encoding utilities
 */

// Base64 encode
export function base64Encode(str: string): string {
  if (typeof window === 'undefined') {
    return str;
  }
  try {
    return btoa(unescape(encodeURIComponent(str)));
  } catch {
    return str;
  }
}

// Base64 decode
export function base64Decode(encoded: string): string {
  if (typeof window === 'undefined') {
    return encoded;
  }
  try {
    return decodeURIComponent(escape(atob(encoded)));
  } catch {
    return encoded;
  }
}

// URL encode
export function urlEncode(str: string): string {
  return encodeURIComponent(str);
}

// URL decode
export function urlDecode(encoded: string): string {
  try {
    return decodeURIComponent(encoded);
  } catch {
    return encoded;
  }
}

// HTML encode
export function htmlEncode(str: string): string {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// HTML decode
export function htmlDecode(encoded: string): string {
  const div = document.createElement('div');
  div.innerHTML = encoded;
  return div.textContent || div.innerText || '';
}

// Encode object to query string
export function encodeQueryString(obj: Record<string, any>): string {
  return Object.entries(obj)
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
    .join('&');
}

// Decode query string to object
export function decodeQueryString(query: string): Record<string, string> {
  const params: Record<string, string> = {};
  const pairs = query.split('&');

  for (const pair of pairs) {
    const [key, value] = pair.split('=');
    if (key) {
      params[decodeURIComponent(key)] = decodeURIComponent(value || '');
    }
  }

  return params;
}



