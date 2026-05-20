export const storage = {
  get: <T,>(key: string, defaultValue: T): T => {
    if (typeof window === 'undefined') {
      return defaultValue;
    }

    try {
      const item = localStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : defaultValue;
    } catch {
      return defaultValue;
    }
  },

  set: <T,>(key: string, value: T): void => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  },

  remove: (key: string): void => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  },

  clear: (): void => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  },

  has: (key: string): boolean => {
    if (typeof window === 'undefined') {
      return false;
    }

    return localStorage.getItem(key) !== null;
  },

  keys: (): string[] => {
    if (typeof window === 'undefined') {
      return [];
    }

    return Object.keys(localStorage);
  },
};

export const sessionStorage = {
  get: <T,>(key: string, defaultValue: T): T => {
    if (typeof window === 'undefined') {
      return defaultValue;
    }

    try {
      const item = window.sessionStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : defaultValue;
    } catch {
      return defaultValue;
    }
  },

  set: <T,>(key: string, value: T): void => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error setting sessionStorage key "${key}":`, error);
    }
  },

  remove: (key: string): void => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      window.sessionStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing sessionStorage key "${key}":`, error);
    }
  },

  clear: (): void => {
    if (typeof window === 'undefined') {
      return;
    }

    try {
      window.sessionStorage.clear();
    } catch (error) {
      console.error('Error clearing sessionStorage:', error);
    }
  },
};



