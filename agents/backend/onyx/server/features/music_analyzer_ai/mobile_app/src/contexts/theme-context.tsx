import React, { createContext, useContext, useMemo } from 'react';
import { useTheme } from '../hooks/use-theme';
import { COLORS } from '../constants/config';

interface ThemeContextValue {
  isDark: boolean;
  colors: typeof COLORS;
  themeMode: 'light' | 'dark' | 'auto';
  toggleTheme: () => void;
  setLightTheme: () => void;
  setDarkTheme: () => void;
  setAutoTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useTheme();

  const colors = useMemo(() => {
    // In a real app, you'd have different color sets for light/dark
    // For now, using the same colors but this structure allows easy extension
    return COLORS;
  }, [theme.isDark]);

  const value: ThemeContextValue = {
    ...theme,
    colors,
  };

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useThemeContext() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useThemeContext must be used within a ThemeProvider');
  }
  return context;
}

