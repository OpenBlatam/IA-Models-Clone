import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { AppState } from '@/types/store';

interface AppStore extends AppState {
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  setLanguage: (language: string) => void;
  setApiBaseUrl: (url: string) => void;
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      theme: 'auto',
      language: 'es',
      apiBaseUrl: 'http://localhost:8000',

      setTheme: (theme) => set({ theme }),
      setLanguage: (language) => set({ language }),
      setApiBaseUrl: (apiBaseUrl) => set({ apiBaseUrl }),
    }),
    {
      name: 'app-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);


