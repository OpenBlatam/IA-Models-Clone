import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useColorScheme } from 'react-native';
import { useStorage } from '../hooks/useStorage';

type Theme = 'light' | 'dark' | 'auto';

interface ThemeColors {
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  primary: string;
  secondary: string;
  error: string;
  success: string;
  warning: string;
  border: string;
  card: string;
  shadow: string;
}

interface ThemeContextType {
  theme: Theme;
  colors: ThemeColors;
  isDark: boolean;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const lightColors: ThemeColors = {
  background: '#ffffff',
  surface: '#f9fafb',
  text: '#1f2937',
  textSecondary: '#6b7280',
  primary: '#6366f1',
  secondary: '#8b5cf6',
  error: '#ef4444',
  success: '#10b981',
  warning: '#f59e0b',
  border: '#e5e7eb',
  card: '#ffffff',
  shadow: 'rgba(0, 0, 0, 0.1)',
};

const darkColors: ThemeColors = {
  background: '#111827',
  surface: '#1f2937',
  text: '#f9fafb',
  textSecondary: '#9ca3af',
  primary: '#818cf8',
  secondary: '#a78bfa',
  error: '#f87171',
  success: '#34d399',
  warning: '#fbbf24',
  border: '#374151',
  card: '#1f2937',
  shadow: 'rgba(0, 0, 0, 0.3)',
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const systemTheme = useColorScheme();
  const [theme, setTheme, , isLoading] = useStorage<Theme>('theme', 'auto');
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    if (theme === 'auto') {
      setIsDark(systemTheme === 'dark');
    } else {
      setIsDark(theme === 'dark');
    }
  }, [theme, systemTheme]);

  const colors = isDark ? darkColors : lightColors;

  const toggleTheme = () => {
    setTheme(isDark ? 'light' : 'dark');
  };

  const value: ThemeContextType = {
    theme,
    colors,
    isDark,
    setTheme,
    toggleTheme,
  };

  if (isLoading) {
    return null; // Or a loading spinner
  }

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

