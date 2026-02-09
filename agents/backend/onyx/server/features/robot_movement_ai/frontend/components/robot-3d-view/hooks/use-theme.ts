/**
 * Hook for theme management
 * @module robot-3d-view/hooks/use-theme
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { getTheme, applyTheme, type ThemeType } from '../lib/themes';

const THEME_STORAGE_KEY = 'robot-3d-view-theme';

/**
 * Hook for managing theme state
 * 
 * @param defaultTheme - Default theme type
 * @returns Theme state and actions
 * 
 * @example
 * ```tsx
 * const { theme, colors, setTheme } = useTheme('dark');
 * ```
 */
export function useTheme(defaultTheme: ThemeType = 'dark') {
  const [theme, setThemeState] = useState<ThemeType>(() => {
    if (typeof window === 'undefined') return defaultTheme;
    
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved && saved in getTheme(defaultTheme)) {
        return saved as ThemeType;
      }
    } catch {
      // Ignore errors
    }
    
    return defaultTheme;
  });

  // Persist theme changes
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore errors
    }
  }, [theme]);

  const setTheme = useCallback((newTheme: ThemeType) => {
    setThemeState(newTheme);
  }, []);

  const colors = useMemo(() => applyTheme(theme), [theme]);

  return {
    theme,
    colors,
    setTheme,
  };
}



