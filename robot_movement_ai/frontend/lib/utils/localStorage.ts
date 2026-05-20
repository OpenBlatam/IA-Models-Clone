// LocalStorage utilities with error handling

export const storage = {
  get<T>(key: string, defaultValue: T | null = null): T | null {
    if (typeof window === 'undefined') return defaultValue;
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return defaultValue;
    }
  },

  set<T>(key: string, value: T): boolean {
    if (typeof window === 'undefined') return false;
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
      return false;
    }
  },

  remove(key: string): boolean {
    if (typeof window === 'undefined') return false;
    try {
      window.localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
      return false;
    }
  },

  clear(): boolean {
    if (typeof window === 'undefined') return false;
    try {
      window.localStorage.clear();
      return true;
    } catch (error) {
      console.error('Error clearing localStorage:', error);
      return false;
    }
  },
};


