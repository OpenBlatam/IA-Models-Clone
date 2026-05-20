interface CookieOptions {
  expires?: Date | number;
  path?: string;
  domain?: string;
  secure?: boolean;
  sameSite?: 'strict' | 'lax' | 'none';
}

export const setCookie = (name: string, value: string, options: CookieOptions = {}): void => {
  if (typeof document === 'undefined') {
    return;
  }

  let cookieString = `${encodeURIComponent(name)}=${encodeURIComponent(value)}`;

  if (options.expires) {
    if (typeof options.expires === 'number') {
      const date = new Date();
      date.setTime(date.getTime() + options.expires * 24 * 60 * 60 * 1000);
      cookieString += `; expires=${date.toUTCString()}`;
    } else {
      cookieString += `; expires=${options.expires.toUTCString()}`;
    }
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
    cookieString += `; sameSite=${options.sameSite}`;
  }

  document.cookie = cookieString;
};

export const getCookie = (name: string): string | null => {
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
};

export const deleteCookie = (name: string, options: Pick<CookieOptions, 'path' | 'domain'> = {}): void => {
  setCookie(name, '', {
    ...options,
    expires: new Date(0),
  });
};

export const getAllCookies = (): Record<string, string> => {
  if (typeof document === 'undefined') {
    return {};
  }

  const cookies: Record<string, string> = {};
  document.cookie.split(';').forEach((cookie) => {
    const [name, value] = cookie.trim().split('=');
    if (name && value) {
      cookies[decodeURIComponent(name)] = decodeURIComponent(value);
    }
  });
  return cookies;
};



