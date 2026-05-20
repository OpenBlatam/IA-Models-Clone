/**
 * Tesla Design System - Exact Design Tokens
 * Based on Tesla's official design specifications
 */

export const teslaColors = {
  // Primary Colors
  black: '#171a20',
  'gray-dark': '#393c41',
  'gray-medium': '#5c5e62',
  'gray-light': '#b5b5b5',
  'gray-lighter': '#e5e7eb',
  white: '#ffffff',
  blue: '#0062cc',
  'blue-hover': '#0052a3',
  'blue-light': '#e6f2ff',
  
  // Semantic Colors
  success: '#10b981',
  'success-light': '#d1fae5',
  error: '#ef4444',
  'error-light': '#fee2e2',
  warning: '#f59e0b',
  'warning-light': '#fef3c7',
  info: '#3b82f6',
  'info-light': '#dbeafe',
} as const;

export const teslaSpacing = {
  xs: '0.5rem',    // 8px
  sm: '0.75rem',  // 12px
  md: '1rem',     // 16px
  lg: '1.5rem',   // 24px
  xl: '2rem',     // 32px
  '2xl': '3rem',  // 48px
  '3xl': '4rem',  // 64px
  '4xl': '6rem',  // 96px
  '5xl': '8rem',  // 128px
} as const;

export const teslaTypography = {
  fontFamily: {
    sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
    display: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
  },
  fontSize: {
    xs: '0.75rem',    // 12px
    sm: '0.875rem',   // 14px
    base: '1rem',     // 16px
    lg: '1.125rem',   // 18px
    xl: '1.25rem',    // 20px
    '2xl': '1.5rem',  // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem', // 36px
    '5xl': '3rem',    // 48px
    '6xl': '3.75rem', // 60px
    '7xl': '4.5rem',  // 72px
    '8xl': '6rem',    // 96px
  },
  fontWeight: {
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  lineHeight: {
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.75,
    loose: 2,
  },
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
  },
} as const;

export const teslaBorderRadius = {
  none: '0',
  xs: '2px',
  sm: '4px',
  md: '6px',
  lg: '8px',
  xl: '12px',
  '2xl': '16px',
  '3xl': '24px',
  full: '9999px',
} as const;

export const teslaShadows = {
  xs: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  sm: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
} as const;

export const teslaTransitions = {
  duration: {
    fast: '150ms',
    base: '200ms',
    slow: '300ms',
    slower: '400ms',
  },
  timing: {
    'ease-in': 'cubic-bezier(0.4, 0, 1, 1)',
    'ease-out': 'cubic-bezier(0, 0, 0.2, 1)',
    'ease-in-out': 'cubic-bezier(0.4, 0, 0.2, 1)',
    spring: 'cubic-bezier(0.16, 1, 0.3, 1)',
  },
} as const;

export const teslaZIndex = {
  base: 0,
  dropdown: 1000,
  sticky: 1020,
  fixed: 1030,
  'modal-backdrop': 1040,
  modal: 1050,
  popover: 1060,
  tooltip: 1070,
} as const;

export const teslaBreakpoints = {
  xs: '475px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
  '3xl': '1920px',
} as const;

export const teslaTouchTargets = {
  minimum: '44px',
  recommended: '48px',
  comfortable: '56px',
} as const;

export const teslaOpacity = {
  0: '0',
  5: '0.05',
  10: '0.1',
  20: '0.2',
  30: '0.3',
  40: '0.4',
  50: '0.5',
  60: '0.6',
  70: '0.7',
  80: '0.8',
  90: '0.9',
  95: '0.95',
  100: '1',
} as const;

export const teslaBlur = {
  none: '0',
  sm: '4px',
  base: '8px',
  md: '12px',
  lg: '16px',
  xl: '20px',
  '2xl': '24px',
  '3xl': '40px',
} as const;

export const teslaBackdropBlur = {
  none: '0',
  sm: '4px',
  base: '8px',
  md: '12px',
  lg: '16px',
  xl: '20px',
  '2xl': '24px',
  '3xl': '40px',
} as const;

export const teslaTransform = {
  scale: {
    '95': '0.95',
    '98': '0.98',
    '100': '1.0',
    '102': '1.02',
    '105': '1.05',
    '110': '1.1',
  },
  translateY: {
    '-1': '-1px',
    '-2': '-2px',
    '-4': '-4px',
    '-8': '-8px',
    '-12': '-12px',
  },
  translateX: {
    '-1': '-1px',
    '-2': '-2px',
    '-4': '-4px',
    '-8': '-8px',
    '1': '1px',
    '2': '2px',
    '4': '4px',
    '8': '8px',
  },
  rotate: {
    '-180': '-180deg',
    '-90': '-90deg',
    '-45': '-45deg',
    '0': '0deg',
    '45': '45deg',
    '90': '90deg',
    '180': '180deg',
  },
} as const;

export const teslaHoverStates = {
  opacity: {
    default: '0.9',
    light: '0.8',
    heavy: '0.95',
  },
  scale: {
    subtle: '1.01',
    normal: '1.02',
    prominent: '1.05',
  },
  translateY: {
    subtle: '-1px',
    normal: '-2px',
    prominent: '-4px',
  },
  shadow: {
    sm: '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  },
} as const;

export const teslaFocusStates = {
  ring: {
    width: '2px',
    offset: '2px',
    color: '#0062cc',
  },
  outline: {
    width: '2px',
    style: 'solid',
    color: '#0062cc',
    offset: '2px',
  },
} as const;

export const teslaActiveStates = {
  scale: {
    pressed: '0.98',
    heavy: '0.95',
  },
  opacity: {
    pressed: '0.8',
  },
} as const;

export const teslaGrid = {
  columns: {
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '6': '6',
    '12': '12',
  },
  gap: {
    xs: '0.5rem',   // 8px
    sm: '0.75rem',  // 12px
    md: '1rem',     // 16px
    lg: '1.5rem',   // 24px
    xl: '2rem',     // 32px
    '2xl': '3rem',  // 48px
  },
} as const;

export const teslaBorderWidth = {
  '0': '0',
  '1': '1px',
  '2': '2px',
  '4': '4px',
  '8': '8px',
} as const;

export const teslaLineHeight = {
  none: '1',
  tight: '1.25',
  snug: '1.375',
  normal: '1.5',
  relaxed: '1.75',
  loose: '2',
  '3': '0.75rem',   // 12px
  '4': '1rem',      // 16px
  '5': '1.25rem',   // 20px
  '6': '1.5rem',    // 24px
  '7': '1.75rem',   // 28px
  '8': '2rem',      // 32px
} as const;

// Helper function to get Tesla color
export function getTeslaColor(color: keyof typeof teslaColors): string {
  return teslaColors[color];
}

// Helper function to get Tesla spacing
export function getTeslaSpacing(size: keyof typeof teslaSpacing): string {
  return teslaSpacing[size];
}

// Helper function to get Tesla shadow
export function getTeslaShadow(size: keyof typeof teslaShadows): string {
  return teslaShadows[size];
}

// Helper function to get Tesla opacity
export function getTeslaOpacity(level: keyof typeof teslaOpacity): string {
  return teslaOpacity[level];
}

// Helper function to get Tesla blur
export function getTeslaBlur(size: keyof typeof teslaBlur): string {
  return teslaBlur[size];
}

// Helper function to get Tesla transform scale
export function getTeslaScale(size: keyof typeof teslaTransform.scale): string {
  return teslaTransform.scale[size];
}

// Helper function to get Tesla transform translateY
export function getTeslaTranslateY(size: keyof typeof teslaTransform.translateY): string {
  return teslaTransform.translateY[size];
}

