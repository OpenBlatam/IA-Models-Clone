import * as EncryptedStorage from 'react-native-encrypted-storage';

const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  USER_DATA: 'user_data',
  SETTINGS: 'app_settings',
} as const;

export interface SecureStorageService {
  setToken: (token: string) => Promise<void>;
  getToken: () => Promise<string | null>;
  removeToken: () => Promise<void>;
  setUserData: (data: unknown) => Promise<void>;
  getUserData: <T>() => Promise<T | null>;
  removeUserData: () => Promise<void>;
  clearAll: () => Promise<void>;
}

async function setItem(key: string, value: string): Promise<void> {
  try {
    await EncryptedStorage.setItem(key, value);
  } catch (error) {
    console.error(`Error setting ${key}:`, error);
    throw error;
  }
}

async function getItem(key: string): Promise<string | null> {
  try {
    return await EncryptedStorage.getItem(key);
  } catch (error) {
    console.error(`Error getting ${key}:`, error);
    return null;
  }
}

async function removeItem(key: string): Promise<void> {
  try {
    await EncryptedStorage.removeItem(key);
  } catch (error) {
    console.error(`Error removing ${key}:`, error);
    throw error;
  }
}

export const secureStorage: SecureStorageService = {
  async setToken(token: string): Promise<void> {
    await setItem(STORAGE_KEYS.AUTH_TOKEN, token);
  },

  async getToken(): Promise<string | null> {
    return await getItem(STORAGE_KEYS.AUTH_TOKEN);
  },

  async removeToken(): Promise<void> {
    await removeItem(STORAGE_KEYS.AUTH_TOKEN);
  },

  async setUserData(data: unknown): Promise<void> {
    await setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(data));
  },

  async getUserData<T>(): Promise<T | null> {
    const data = await getItem(STORAGE_KEYS.USER_DATA);
    if (!data) return null;
    try {
      return JSON.parse(data) as T;
    } catch {
      return null;
    }
  },

  async removeUserData(): Promise<void> {
    await removeItem(STORAGE_KEYS.USER_DATA);
  },

  async clearAll(): Promise<void> {
    try {
      await EncryptedStorage.clear();
    } catch (error) {
      console.error('Error clearing storage:', error);
      throw error;
    }
  },
};

