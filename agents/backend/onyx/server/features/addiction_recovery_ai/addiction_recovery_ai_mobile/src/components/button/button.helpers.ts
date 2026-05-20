// Helper functions for Button component (pure functions)

import type { ColorScheme } from '@/theme/colors';

export function getButtonBackgroundColor(
  variant: 'primary' | 'secondary' | 'outline',
  colors: ColorScheme
): string | 'transparent' {
  if (variant === 'primary') return colors.primary;
  if (variant === 'secondary') return colors.secondary;
  return 'transparent';
}

export function getButtonTextColor(
  variant: 'primary' | 'secondary' | 'outline',
  colors: ColorScheme
): string {
  if (variant === 'outline') return colors.primary;
  return '#FFFFFF';
}

export function getButtonBorderColor(
  variant: 'primary' | 'secondary' | 'outline',
  colors: ColorScheme
): string | undefined {
  if (variant === 'outline') return colors.primary;
  return undefined;
}

