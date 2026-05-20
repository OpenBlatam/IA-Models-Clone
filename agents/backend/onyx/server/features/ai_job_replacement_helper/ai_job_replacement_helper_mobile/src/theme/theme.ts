import { useColorScheme } from 'react-native';
import { lightColors, darkColors, type ColorScheme } from './colors';

export interface Theme {
  colors: ColorScheme;
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
    xxl: number;
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
    full: number;
  };
  typography: {
    h1: { fontSize: number; fontWeight: string; lineHeight: number };
    h2: { fontSize: number; fontWeight: string; lineHeight: number };
    h3: { fontSize: number; fontWeight: string; lineHeight: number };
    body: { fontSize: number; fontWeight: string; lineHeight: number };
    caption: { fontSize: number; fontWeight: string; lineHeight: number };
    button: { fontSize: number; fontWeight: string; lineHeight: number };
  };
  shadows: {
    sm: object;
    md: object;
    lg: object;
  };
}

const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

const borderRadius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  full: 9999,
};

const typography = {
  h1: { fontSize: 32, fontWeight: 'bold', lineHeight: 40 },
  h2: { fontSize: 24, fontWeight: 'bold', lineHeight: 32 },
  h3: { fontSize: 20, fontWeight: '600', lineHeight: 28 },
  body: { fontSize: 16, fontWeight: '400', lineHeight: 24 },
  caption: { fontSize: 14, fontWeight: '400', lineHeight: 20 },
  button: { fontSize: 16, fontWeight: '600', lineHeight: 24 },
};

const shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 8,
  },
};

export const lightTheme: Theme = {
  colors: lightColors,
  spacing,
  borderRadius,
  typography,
  shadows,
};

export const darkTheme: Theme = {
  colors: darkColors,
  spacing,
  borderRadius,
  typography,
  shadows,
};

export function useTheme(): Theme {
  const colorScheme = useColorScheme();
  return colorScheme === 'dark' ? darkTheme : lightTheme;
}


