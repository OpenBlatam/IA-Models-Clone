export const lightColors = {
  primary: '#007AFF',
  secondary: '#4ECDC4',
  accent: '#FFD700',
  background: '#FFFFFF',
  surface: '#F5F5F5',
  text: '#333333',
  textSecondary: '#666666',
  textTertiary: '#999999',
  border: '#E0E0E0',
  error: '#FF3B30',
  success: '#34C759',
  warning: '#FF9500',
  info: '#007AFF',
  card: '#FFFFFF',
  notification: '#FF3B30',
  disabled: '#CCCCCC',
  placeholder: '#999999',
} as const;

export const darkColors = {
  primary: '#0A84FF',
  secondary: '#64D2FF',
  accent: '#FFD60A',
  background: '#000000',
  surface: '#1C1C1E',
  text: '#FFFFFF',
  textSecondary: '#EBEBF5',
  textTertiary: '#8E8E93',
  border: '#38383A',
  error: '#FF453A',
  success: '#30D158',
  warning: '#FF9F0A',
  info: '#0A84FF',
  card: '#1C1C1E',
  notification: '#FF453A',
  disabled: '#3A3A3C',
  placeholder: '#8E8E93',
} as const;

export type ColorScheme = typeof lightColors;


