/**
 * Enhanced storage utilities with encryption support
 */

import { storage } from './localStorage';

export interface StorageOptions {
  encrypt?: boolean;
  ttl?: number; // Time to live in milliseconds
}

class EnhancedStorage {
  private encryptionKey = 'robot-movement-ai-key'; // In production, use environment variable

  private encrypt(data: string): string {
    // Simple encryption (in production, use proper encryption)
    if (typeof window === 'undefined') {
      return data;
    }
    // For now, just base64 encode (not secure, but obfuscates)
    return btoa(data);
  }

  private decrypt(encryptedData: string): string {
    if (typeof window === 'undefined') {
      return encryptedData;
    }
    try {
      return atob(encryptedData);
    } catch {
      return encryptedData;
    }
  }

  set<T>(key: string, value: T, options: StorageOptions = {}): boolean {
    const { encrypt = false, ttl } = options;

    try {
      const data = {
        value,
        timestamp: Date.now(),
        expiresAt: ttl ? Date.now() + ttl : null,
      };

      let serialized = JSON.stringify(data);
      if (encrypt) {
        serialized = this.encrypt(serialized);
      }

      return storage.set(key, serialized);
    } catch (error) {
      console.error(`Error setting storage key "${key}":`, error);
      return false;
    }
  }

  get<T>(key: string, defaultValue: T | null = null, encrypted: boolean = false): T | null {
    try {
      const raw = storage.get<string>(key, null);
      if (!raw) {
        return defaultValue;
      }

      let data: { value: T; timestamp: number; expiresAt: number | null };
      try {
        const parsed = encrypted ? this.decrypt(raw) : raw;
        data = JSON.parse(parsed);
      } catch {
        // If decryption/parsing fails, try as plain value (backward compatibility)
        return (raw as unknown) as T;
      }

      // Check expiration
      if (data.expiresAt && Date.now() > data.expiresAt) {
        this.delete(key);
        return defaultValue;
      }

      return data.value;
    } catch (error) {
      console.error(`Error getting storage key "${key}":`, error);
      return defaultValue;
    }
  }

  delete(key: string): boolean {
    return storage.remove(key);
  }

  clear(): boolean {
    return storage.clear();
  }

  // Get all keys
  getAllKeys(): string[] {
    if (typeof window === 'undefined') {
      return [];
    }

    const keys: string[] = [];
    for (let i = 0; i < window.localStorage.length; i++) {
      const key = window.localStorage.key(i);
      if (key) {
        keys.push(key);
      }
    }
    return keys;
  }

  // Get storage size (approximate)
  getSize(): number {
    if (typeof window === 'undefined') {
      return 0;
    }

    let total = 0;
    for (const key in window.localStorage) {
      if (window.localStorage.hasOwnProperty(key)) {
        total += window.localStorage[key].length + key.length;
      }
    }
    return total;
  }
}

export const enhancedStorage = new EnhancedStorage();



