/**
 * Cookie utility functions.
 * Provides helper functions for cookie manipulation.
 */

/**
 * Sets a cookie.
 * @param name - Cookie name
 * @param value - Cookie value
 * @param options - Cookie options
 */
export function setCookie(
  name: string,
  value: string,
  options: {
    expires?: Date | number;
    path?: string;
    domain?: string;
    secure?: boolean;
    sameSite?: 'strict' | 'lax' | 'none';
  } = {}
): void {
  if (typeof document === 'undefined') {
    return;
  }

  let cookieString = `${encodeURIComponent(name)}=${encodeURIComponent(value)}`;

  if (options.expires) {
    const expiresDate =
      options.expires instanceof Date
        ? options.expires
        : new Date(Date.now() + options.expires * 24 * 60 * 60 * 1000);
    cookieString += `; expires=${expiresDate.toUTCString()}`;
  }

  if (options.path) {
    cookieString += `; path=${options.path}`;
  }

  if (options.domain) {
    cookieString += `; domain=${options.domain}`;
  }

  if (options.secure) {
    cookieString += '; secure';
  }

  if (options.sameSite) {
    cookieString += `; samesite=${options.sameSite}`;
  }

  document.cookie = cookieString;
}

/**
 * Gets a cookie value.
 * @param name - Cookie name
 * @returns Cookie value or null
 */
export function getCookie(name: string): string | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const nameEQ = `${encodeURIComponent(name)}=`;
  const cookies = document.cookie.split(';');

  for (let i = 0; i < cookies.length; i++) {
    let cookie = cookies[i];
    while (cookie.charAt(0) === ' ') {
      cookie = cookie.substring(1, cookie.length);
    }
    if (cookie.indexOf(nameEQ) === 0) {
      return decodeURIComponent(cookie.substring(nameEQ.length, cookie.length));
    }
  }

  return null;
}

/**
 * Removes a cookie.
 * @param name - Cookie name
 * @param options - Cookie options (path, domain)
 */
export function removeCookie(
  name: string,
  options: { path?: string; domain?: string } = {}
): void {
  setCookie(name, '', {
    ...options,
    expires: new Date(0),
  });
}

/**
 * Gets all cookies as an object.
 * @returns Object with all cookies
 */
export function getAllCookies(): Record<string, string> {
  if (typeof document === 'undefined') {
    return {};
  }

  const cookies: Record<string, string> = {};
  const cookieStrings = document.cookie.split(';');

  for (const cookieString of cookieStrings) {
    const [name, value] = cookieString.trim().split('=');
    if (name && value) {
      cookies[decodeURIComponent(name)] = decodeURIComponent(value);
    }
  }

  return cookies;
}

