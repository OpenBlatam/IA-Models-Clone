/**
 * App Context
 * ===========
 * Global app context for theme, settings, etc.
 */

import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { useColorScheme } from 'react-native';
import { Colors } from '@/constants/colors';

interface AppState {
  theme: 'light' | 'dark' | 'auto';
  isDark: boolean;
  colors: typeof Colors.light | typeof Colors.dark;
}

type AppAction = { type: 'SET_THEME'; payload: 'light' | 'dark' | 'auto' };

const initialState: AppState = {
  theme: 'auto',
  isDark: false,
  colors: Colors.light,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_THEME':
      return {
        ...state,
        theme: action.payload,
      };
    default:
      return state;
  }
}

interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const systemColorScheme = useColorScheme();
  const [state, dispatch] = useReducer(appReducer, initialState);

  const isDark =
    state.theme === 'dark' || (state.theme === 'auto' && systemColorScheme === 'dark');

  const colors = isDark ? Colors.dark : Colors.light;

  const setTheme = (theme: 'light' | 'dark' | 'auto') => {
    dispatch({ type: 'SET_THEME', payload: theme });
  };

  const value: AppContextValue = {
    state: { ...state, isDark, colors },
    dispatch,
    setTheme,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}




