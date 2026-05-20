import AsyncStorage from '@react-native-async-storage/async-storage';
import EncryptedStorage from 'react-native-encrypted-storage';

const STORAGE_KEYS = {
  FAVORITES: '@music_analyzer:favorites',
  RECENT_SEARCHES: '@music_analyzer:recent_searches',
  USER_PREFERENCES: '@music_analyzer:user_preferences',
} as const;

export async function saveFavorites(tracks: unknown[]): Promise<void> {
  try {
    await AsyncStorage.setItem(
      STORAGE_KEYS.FAVORITES,
      JSON.stringify(tracks)
    );
  } catch (error) {
    console.error('Error saving favorites:', error);
  }
}

export async function loadFavorites(): Promise<unknown[]> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.FAVORITES);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('Error loading favorites:', error);
    return [];
  }
}

export async function saveRecentSearches(tracks: unknown[]): Promise<void> {
  try {
    await AsyncStorage.setItem(
      STORAGE_KEYS.RECENT_SEARCHES,
      JSON.stringify(tracks)
    );
  } catch (error) {
    console.error('Error saving recent searches:', error);
  }
}

export async function loadRecentSearches(): Promise<unknown[]> {
  try {
    const data = await AsyncStorage.getItem(STORAGE_KEYS.RECENT_SEARCHES);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('Error loading recent searches:', error);
    return [];
  }
}

export async function saveSecureData(key: string, value: string): Promise<void> {
  try {
    await EncryptedStorage.setItem(key, value);
  } catch (error) {
    console.error('Error saving secure data:', error);
  }
}

export async function loadSecureData(key: string): Promise<string | null> {
  try {
    return await EncryptedStorage.getItem(key);
  } catch (error) {
    console.error('Error loading secure data:', error);
    return null;
  }
}

