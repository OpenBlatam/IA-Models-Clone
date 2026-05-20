// Helper functions for Input component (pure functions)

import type { ColorScheme } from '@/theme/colors';

export function getInputBorderColor(
  hasError: boolean,
  colors: ColorScheme
): string {
  return hasError ? colors.error : colors.border;
}

export function getInputBorderWidth(hasError: boolean): number {
  return hasError ? 2 : 1;
}

