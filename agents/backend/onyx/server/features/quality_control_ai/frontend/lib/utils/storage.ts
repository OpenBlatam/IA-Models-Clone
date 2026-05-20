export const setLocalStorage = (key: string, value: unknown): boolean => {
  try {
    const serialized = JSON.stringify(value);
    localStorage.setItem(key, serialized);
    return true;
  } catch (error) {
    console.error('Error setting localStorage:', error);
    return false;
  }
};

export const getLocalStorage = <T,>(key: string, defaultValue?: T): T | null => {
  try {
    const item = localStorage.getItem(key);
    if (item === null) return defaultValue ?? null;
    return JSON.parse(item) as T;
  } catch (error) {
    console.error('Error getting localStorage:', error);
    return defaultValue ?? null;
  }
};

export const removeLocalStorage = (key: string): boolean => {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error removing localStorage:', error);
    return false;
  }
};

export const clearLocalStorage = (): boolean => {
  try {
    localStorage.clear();
    return true;
  } catch (error) {
    console.error('Error clearing localStorage:', error);
    return false;
  }
};

export const setSessionStorage = (key: string, value: unknown): boolean => {
  try {
    const serialized = JSON.stringify(value);
    sessionStorage.setItem(key, serialized);
    return true;
  } catch (error) {
    console.error('Error setting sessionStorage:', error);
    return false;
  }
};

export const getSessionStorage = <T,>(key: string, defaultValue?: T): T | null => {
  try {
    const item = sessionStorage.getItem(key);
    if (item === null) return defaultValue ?? null;
    return JSON.parse(item) as T;
  } catch (error) {
    console.error('Error getting sessionStorage:', error);
    return defaultValue ?? null;
  }
};

export const removeSessionStorage = (key: string): boolean => {
  try {
    sessionStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error removing sessionStorage:', error);
    return false;
  }
};

export const clearSessionStorage = (): boolean => {
  try {
    sessionStorage.clear();
    return true;
  } catch (error) {
    console.error('Error clearing sessionStorage:', error);
    return false;
  }
};

