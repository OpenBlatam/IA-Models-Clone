/**
 * Color utilities
 */

// Convert hex to RGB
export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

// Convert RGB to hex
export function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('');
}

// Get contrast color (black or white)
export function getContrastColor(hex: string): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return '#000000';

  // Calculate relative luminance
  const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
  return luminance > 0.5 ? '#000000' : '#ffffff';
}

// Lighten color
export function lighten(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;

  const r = Math.min(255, rgb.r + (255 - rgb.r) * (percent / 100));
  const g = Math.min(255, rgb.g + (255 - rgb.g) * (percent / 100));
  const b = Math.min(255, rgb.b + (255 - rgb.b) * (percent / 100));

  return rgbToHex(Math.round(r), Math.round(g), Math.round(b));
}

// Darken color
export function darken(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;

  const r = Math.max(0, rgb.r * (1 - percent / 100));
  const g = Math.max(0, rgb.g * (1 - percent / 100));
  const b = Math.max(0, rgb.b * (1 - percent / 100));

  return rgbToHex(Math.round(r), Math.round(g), Math.round(b));
}

// Add alpha to hex color
export function addAlpha(hex: string, alpha: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;

  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}



