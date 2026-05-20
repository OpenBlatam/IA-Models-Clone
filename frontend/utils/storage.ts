export class StorageManager {
  private prefix: string;

  constructor(prefix: string = 'bul_') {
    this.prefix = prefix;
  }

  private getKey(key: string): string {
    return `${this.prefix}${key}`;
  }

  get<T>(key: string, defaultValue?: T): T | null {
    if (typeof window === 'undefined') return defaultValue || null;
    
    try {
      const item = localStorage.getItem(this.getKey(key));
      if (item === null) return defaultValue || null;
      return JSON.parse(item) as T;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return defaultValue || null;
    }
  }

  set<T>(key: string, value: T): void {
    if (typeof window === 'undefined') return;
    
    try {
      localStorage.setItem(this.getKey(key), JSON.stringify(value));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }

  remove(key: string): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem(this.getKey(key));
  }

  clear(): void {
    if (typeof window === 'undefined') return;
    
    const keys = Object.keys(localStorage);
    keys.forEach((key) => {
      if (key.startsWith(this.prefix)) {
        localStorage.removeItem(key);
      }
    });
  }

  getAll(): Record<string, any> {
    if (typeof window === 'undefined') return {};
    
    const result: Record<string, any> = {};
    const keys = Object.keys(localStorage);
    
    keys.forEach((key) => {
      if (key.startsWith(this.prefix)) {
        const cleanKey = key.replace(this.prefix, '');
        try {
          result[cleanKey] = JSON.parse(localStorage.getItem(key) || 'null');
        } catch {
          result[cleanKey] = localStorage.getItem(key);
        }
      }
    });
    
    return result;
  }

  has(key: string): boolean {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem(this.getKey(key)) !== null;
  }
}

export const storage = new StorageManager();

