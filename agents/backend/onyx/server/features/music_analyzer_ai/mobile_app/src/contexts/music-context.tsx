import React, { createContext, useContext, useReducer, useCallback } from 'react';
import type { Track, TrackAnalysis } from '../types/api';

interface MusicState {
  recentSearches: Track[];
  currentAnalysis: TrackAnalysis | null;
  favorites: Track[];
  isLoading: boolean;
  error: string | null;
}

type MusicAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CURRENT_ANALYSIS'; payload: TrackAnalysis | null }
  | { type: 'ADD_RECENT_SEARCH'; payload: Track }
  | { type: 'ADD_FAVORITE'; payload: Track }
  | { type: 'REMOVE_FAVORITE'; payload: string }
  | { type: 'CLEAR_RECENT_SEARCHES' };

interface MusicContextValue extends MusicState {
  addRecentSearch: (track: Track) => void;
  addFavorite: (track: Track) => void;
  removeFavorite: (trackId: string) => void;
  clearRecentSearches: () => void;
  setCurrentAnalysis: (analysis: TrackAnalysis | null) => void;
}

const initialState: MusicState = {
  recentSearches: [],
  currentAnalysis: null,
  favorites: [],
  isLoading: false,
  error: null,
};

function musicReducer(
  state: MusicState,
  action: MusicAction
): MusicState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'SET_CURRENT_ANALYSIS':
      return { ...state, currentAnalysis: action.payload };
    case 'ADD_RECENT_SEARCH': {
      const exists = state.recentSearches.some(
        (t) => t.id === action.payload.id
      );
      if (exists) return state;
      return {
        ...state,
        recentSearches: [action.payload, ...state.recentSearches].slice(0, 10),
      };
    }
    case 'ADD_FAVORITE': {
      const exists = state.favorites.some((t) => t.id === action.payload.id);
      if (exists) return state;
      return { ...state, favorites: [...state.favorites, action.payload] };
    }
    case 'REMOVE_FAVORITE':
      return {
        ...state,
        favorites: state.favorites.filter((t) => t.id !== action.payload),
      };
    case 'CLEAR_RECENT_SEARCHES':
      return { ...state, recentSearches: [] };
    default:
      return state;
  }
}

const MusicContext = createContext<MusicContextValue | undefined>(undefined);

export function MusicProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(musicReducer, initialState);

  const addRecentSearch = useCallback((track: Track) => {
    dispatch({ type: 'ADD_RECENT_SEARCH', payload: track });
  }, []);

  const addFavorite = useCallback((track: Track) => {
    dispatch({ type: 'ADD_FAVORITE', payload: track });
  }, []);

  const removeFavorite = useCallback((trackId: string) => {
    dispatch({ type: 'REMOVE_FAVORITE', payload: trackId });
  }, []);

  const clearRecentSearches = useCallback(() => {
    dispatch({ type: 'CLEAR_RECENT_SEARCHES' });
  }, []);

  const setCurrentAnalysis = useCallback(
    (analysis: TrackAnalysis | null) => {
      dispatch({ type: 'SET_CURRENT_ANALYSIS', payload: analysis });
    },
    []
  );

  const value: MusicContextValue = {
    ...state,
    addRecentSearch,
    addFavorite,
    removeFavorite,
    clearRecentSearches,
    setCurrentAnalysis,
  };

  return (
    <MusicContext.Provider value={value}>{children}</MusicContext.Provider>
  );
}

export function useMusic() {
  const context = useContext(MusicContext);
  if (context === undefined) {
    throw new Error('useMusic must be used within a MusicProvider');
  }
  return context;
}

