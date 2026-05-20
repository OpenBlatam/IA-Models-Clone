import React, { createContext, useContext, useEffect } from 'react';
import { useColorScheme as useRNColorScheme } from 'react-native';
import { useAppStore } from '@/store/app-store';
import { StatusBar } from 'expo-status-bar';

interface ThemeContextValue {
  theme: 'light' | 'dark';
  isDark: boolean;
  colors: {
    background: string;
    foreground: string;
    primary: string;
    secondary: string;
    border: string;
    card: string;
    text: string;
    textSecondary: string;
    error: string;
    success: string;
    warning: string;
  };
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
}

const lightColors = {
  background: '#FFFFFF',
  foreground: '#000000',
  primary: '#007AFF',
  secondary: '#5856D6',
  border: '#C7C7CC',
  card: '#F5F5F5',
  text: '#000000',
  textSecondary: '#666666',
  error: '#FF3B30',
  success: '#34C759',
  warning: '#FF9500',
};

const darkColors = {
  background: '#000000',
  foreground: '#FFFFFF',
  primary: '#0A84FF',
  secondary: '#5E5CE6',
  border: '#38383A',
  card: '#1C1C1E',
  text: '#FFFFFF',
  textSecondary: '#8E8E93',
  error: '#FF453A',
  success: '#32D74B',
  warning: '#FF9F0A',
};

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemColorScheme = useRNColorScheme();
  const { theme: storedTheme, setTheme: setStoredTheme } = useAppStore();

  const effectiveTheme =
    storedTheme === 'auto' ? systemColorScheme || 'light' : storedTheme;

  const isDark = effectiveTheme === 'dark';
  const colors = isDark ? darkColors : lightColors;

  useEffect(() => {
    // Update status bar style based on theme
  }, [isDark]);

  const setTheme = (theme: 'light' | 'dark' | 'auto') => {
    setStoredTheme(theme);
  };

  const value: ThemeContextValue = {
    theme: effectiveTheme,
    isDark,
    colors,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}


