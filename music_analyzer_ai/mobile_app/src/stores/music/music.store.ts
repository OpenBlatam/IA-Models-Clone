import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { Track, TrackAnalysis } from '../../types/api';

interface MusicState {
  recentSearches: Track[];
  currentAnalysis: TrackAnalysis | null;
  favorites: Track[];
  isLoading: boolean;
  error: string | null;
}

interface MusicActions {
  addRecentSearch: (track: Track) => void;
  addFavorite: (track: Track) => void;
  removeFavorite: (trackId: string) => void;
  clearRecentSearches: () => void;
  setCurrentAnalysis: (analysis: TrackAnalysis | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

type MusicStore = MusicState & MusicActions;

const initialState: MusicState = {
  recentSearches: [],
  currentAnalysis: null,
  favorites: [],
  isLoading: false,
  error: null,
};

export const useMusicStore = create<MusicStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      addRecentSearch: (track: Track) => {
        const { recentSearches } = get();
        const exists = recentSearches.some((t) => t.id === track.id);
        if (exists) return;

        set({
          recentSearches: [track, ...recentSearches].slice(0, 10),
        });
      },

      addFavorite: (track: Track) => {
        const { favorites } = get();
        const exists = favorites.some((t) => t.id === track.id);
        if (exists) return;

        set({
          favorites: [...favorites, track],
        });
      },

      removeFavorite: (trackId: string) => {
        set({
          favorites: get().favorites.filter((t) => t.id !== trackId),
        });
      },

      clearRecentSearches: () => {
        set({ recentSearches: [] });
      },

      setCurrentAnalysis: (analysis: TrackAnalysis | null) => {
        set({ currentAnalysis: analysis });
      },

      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },

      setError: (error: string | null) => {
        set({ error });
      },

      reset: () => {
        set(initialState);
      },
    }),
    {
      name: 'music-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        favorites: state.favorites,
        recentSearches: state.recentSearches,
      }),
    }
  )
);

export const useMusic = () => useMusicStore();


