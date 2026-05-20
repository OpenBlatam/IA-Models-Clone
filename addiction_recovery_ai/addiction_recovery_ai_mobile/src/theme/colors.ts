import { useColorScheme } from 'react-native';

export interface ColorScheme {
  primary: string;
  secondary: string;
  success: string;
  warning: string;
  error: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  card: string;
  shadow: string;
}

const lightColors: ColorScheme = {
  primary: '#007AFF',
  secondary: '#5856D6',
  success: '#34C759',
  warning: '#FF9500',
  error: '#FF3B30',
  background: '#F5F5F5',
  surface: '#FFFFFF',
  text: '#000000',
  textSecondary: '#666666',
  border: '#E5E5EA',
  card: '#FFFFFF',
  shadow: '#000000',
};

const darkColors: ColorScheme = {
  primary: '#0A84FF',
  secondary: '#5E5CE6',
  success: '#30D158',
  warning: '#FF9F0A',
  error: '#FF453A',
  background: '#000000',
  surface: '#1C1C1E',
  text: '#FFFFFF',
  textSecondary: '#98989D',
  border: '#38383A',
  card: '#1C1C1E',
  shadow: '#000000',
};

export function getColors(colorScheme: 'light' | 'dark' | null): ColorScheme {
  return colorScheme === 'dark' ? darkColors : lightColors;
}

export function useColors(): ColorScheme {
  const colorScheme = useColorScheme();
  return getColors(colorScheme);
}

