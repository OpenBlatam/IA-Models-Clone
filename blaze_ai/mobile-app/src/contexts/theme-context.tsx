import React, { createContext, useContext, useEffect, useState } from 'react';
import { useColorScheme } from 'react-native';
import { Theme, getTheme } from '@/constants/theme';

interface ThemeContextType {
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
  setTheme: (isDark: boolean) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps): JSX.Element {
  const colorScheme = useColorScheme();
  const [isDark, setIsDark] = useState(colorScheme === 'dark');
  const [theme, setThemeState] = useState<Theme>(getTheme(isDark));

  useEffect(() => {
    const newTheme = getTheme(isDark);
    setThemeState(newTheme);
  }, [isDark]);

  useEffect(() => {
    if (colorScheme) {
      setIsDark(colorScheme === 'dark');
    }
  }, [colorScheme]);

  function toggleTheme(): void {
    setIsDark(prev => !prev);
  }

  function setTheme(darkMode: boolean): void {
    setIsDark(darkMode);
  }

  const value: ThemeContextType = {
    theme,
    isDark,
    toggleTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}
import { useColorScheme } from 'react-native';
import { Theme, getTheme } from '@/constants/theme';

interface ThemeContextType {
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
  setTheme: (isDark: boolean) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps): JSX.Element {
  const colorScheme = useColorScheme();
  const [isDark, setIsDark] = useState(colorScheme === 'dark');
  const [theme, setThemeState] = useState<Theme>(getTheme(isDark));

  useEffect(() => {
    const newTheme = getTheme(isDark);
    setThemeState(newTheme);
  }, [isDark]);

  useEffect(() => {
    if (colorScheme) {
      setIsDark(colorScheme === 'dark');
    }
  }, [colorScheme]);

  function toggleTheme(): void {
    setIsDark(prev => !prev);
  }

  function setTheme(darkMode: boolean): void {
    setIsDark(darkMode);
  }

  const value: ThemeContextType = {
    theme,
    isDark,
    toggleTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}


