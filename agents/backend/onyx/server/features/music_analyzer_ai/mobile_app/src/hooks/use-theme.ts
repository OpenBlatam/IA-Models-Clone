import { useState, useEffect, useCallback } from 'react';
import { useColorScheme } from 'react-native';
import { useLocalStorage } from './use-local-storage';

type ThemeMode = 'light' | 'dark' | 'auto';

interface ThemeColors {
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  error: string;
  success: string;
  warning: string;
  info: string;
}

/**
 * Hook for theme management
 * Supports light, dark, and auto modes
 */
export function useTheme() {
  const systemColorScheme = useColorScheme();
  const [themeMode, setThemeMode] = useLocalStorage<ThemeMode>('theme-mode', 'auto');
  const [isDark, setIsDark] = useState(systemColorScheme === 'dark');

  useEffect(() => {
    if (themeMode === 'auto') {
      setIsDark(systemColorScheme === 'dark');
    } else {
      setIsDark(themeMode === 'dark');
    }
  }, [themeMode, systemColorScheme]);

  const toggleTheme = useCallback(() => {
    if (themeMode === 'auto') {
      setThemeMode('light');
    } else if (themeMode === 'light') {
      setThemeMode('dark');
    } else {
      setThemeMode('auto');
    }
  }, [themeMode, setThemeMode]);

  const setLightTheme = useCallback(() => {
    setThemeMode('light');
  }, [setThemeMode]);

  const setDarkTheme = useCallback(() => {
    setThemeMode('dark');
  }, [setThemeMode]);

  const setAutoTheme = useCallback(() => {
    setThemeMode('auto');
  }, [setThemeMode]);

  return {
    isDark,
    themeMode,
    toggleTheme,
    setLightTheme,
    setDarkTheme,
    setAutoTheme,
  };
}

