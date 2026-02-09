/**
 * Color utility functions
 * @module robot-3d-view/utils/color-utils
 */

/**
 * Converts hex color to RGB
 * 
 * @param hex - Hex color string
 * @returns RGB values [r, g, b] or null if invalid
 */
export function hexToRgb(hex: string): [number, number, number] | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16) / 255,
        parseInt(result[2], 16) / 255,
        parseInt(result[3], 16) / 255,
      ]
    : null;
}

/**
 * Interpolates between two colors
 * 
 * @param color1 - First color (hex)
 * @param color2 - Second color (hex)
 * @param factor - Interpolation factor (0-1)
 * @returns Interpolated hex color
 */
export function interpolateColor(color1: string, color2: string, factor: number): string {
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);

  if (!rgb1 || !rgb2) return color1;

  const r = Math.round(rgb1[0] * 255 + (rgb2[0] * 255 - rgb1[0] * 255) * factor);
  const g = Math.round(rgb1[1] * 255 + (rgb2[1] * 255 - rgb1[1] * 255) * factor);
  const b = Math.round(rgb1[2] * 255 + (rgb2[2] * 255 - rgb1[2] * 255) * factor);

  return `#${[r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Gets status color based on robot status
 * 
 * @param status - Robot status
 * @returns Hex color string
 */
export function getStatusColor(status: 'idle' | 'moving' | 'error' | 'connected' | 'disconnected'): string {
  const colors = {
    idle: '#3b82f6',
    moving: '#10b981',
    error: '#ef4444',
    connected: '#10b981',
    disconnected: '#ef4444',
  };
  return colors[status] || colors.idle;
}



