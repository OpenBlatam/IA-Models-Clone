import React, { createContext, useContext, useMemo, ReactNode } from 'react';
import { useColorScheme } from 'react-native';
import { getColors, type ColorScheme } from '@/theme/colors';
import { getShadows } from '@/theme/shadows';
import { SPACING } from '@/theme/spacing';
import { TYPOGRAPHY } from '@/theme/typography';

interface ThemeContextValue {
  colors: ColorScheme;
  shadows: ReturnType<typeof getShadows>;
  spacing: typeof SPACING;
  typography: typeof TYPOGRAPHY;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps): JSX.Element {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const colors = useMemo(() => getColors(colorScheme), [colorScheme]);
  const shadows = useMemo(() => getShadows(colors), [colors]);

  const value = useMemo(
    () => ({
      colors,
      shadows,
      spacing: SPACING,
      typography: TYPOGRAPHY,
      isDark,
    }),
    [colors, shadows, isDark]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}

