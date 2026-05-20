/**
 * Tesla Exact Border Values
 * Based on Tesla's exact border system
 */

export const teslaExactBorders = {
  // Border Width (exact)
  width: {
    '0': '0',
    '1': '1px',
    '2': '2px',
    '4': '4px',
    '8': '8px',
  },
  
  // Border Radius (exact)
  radius: {
    none: '0',
    xs: '2px',
    sm: '4px',
    md: '6px',
    lg: '8px',
    xl: '12px',
    '2xl': '16px',
    '3xl': '24px',
    full: '9999px',
  },
  
  // Border Colors (exact)
  color: {
    default: '#e5e7eb',      // gray-200
    light: '#f3f4f6',         // gray-100
    dark: '#d1d5db',          // gray-300
    focus: '#0062cc',         // tesla-blue
    error: '#ef4444',         // red-500
    success: '#10b981',       // green-500
    warning: '#f59e0b',       // yellow-500
  },
  
  // Component Specific Borders
  card: {
    width: '1px',
    color: '#e5e7eb',
    radius: '8px',
  },
  
  button: {
    width: {
      primary: '0',
      secondary: '1px',
      outline: '2px',
    },
    color: {
      secondary: '#d1d5db',
      outline: '#393c41',
    },
    radius: '4px',
  },
  
  input: {
    width: '1px',
    color: {
      default: '#d1d5db',
      focus: '#0062cc',
      error: '#ef4444',
    },
    radius: '4px',
  },
  
  modal: {
    width: '1px',
    color: '#e5e7eb',
    radius: '8px',
  },
} as const;

// Helper functions
export function getTeslaBorderWidth(size: keyof typeof teslaExactBorders.width): string {
  return teslaExactBorders.width[size];
}

export function getTeslaBorderRadius(size: keyof typeof teslaExactBorders.radius): string {
  return teslaExactBorders.radius[size];
}

export function getTeslaBorderColor(color: keyof typeof teslaExactBorders.color): string {
  return teslaExactBorders.color[color];
}



