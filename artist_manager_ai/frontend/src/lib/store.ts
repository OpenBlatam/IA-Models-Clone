import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface AppState {
  artistId: string;
  setArtistId: (id: string) => void;
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      artistId: 'artist_001',
      setArtistId: (id) => set({ artistId: id }),
      theme: 'light',
      setTheme: (theme) => set({ theme }),
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
    }),
    {
      name: 'artist-manager-storage',
      storage: createJSONStorage(() => localStorage),
    }
  )
);

