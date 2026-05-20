import { PixelRatio } from 'react-native';

/**
 * Text scaling utilities for accessibility
 * Ensures text scales properly with system font size settings
 */

/**
 * Get scaled font size based on system font scale
 */
export function getScaledFontSize(baseSize: number): number {
  const fontScale = PixelRatio.getFontScale();
  return Math.round(baseSize * fontScale);
}

/**
 * Get scaled dimensions based on screen density
 */
export function getScaledSize(size: number): number {
  return PixelRatio.getPixelSizeForLayoutSize(size);
}

/**
 * Check if user has increased font size
 */
export function hasIncreasedFontSize(): boolean {
  return PixelRatio.getFontScale() > 1.0;
}

/**
 * Get accessibility-friendly font sizes
 */
export const ACCESSIBLE_FONT_SIZES = {
  small: getScaledFontSize(14),
  medium: getScaledFontSize(16),
  large: getScaledFontSize(18),
  xlarge: getScaledFontSize(20),
} as const;

/**
 * Minimum touch target size for accessibility (44x44 points)
 */
export const MIN_TOUCH_TARGET = 44;

/**
 * Check if size meets minimum touch target
 */
export function meetsMinimumTouchTarget(size: number): boolean {
  return size >= MIN_TOUCH_TARGET;
}

