/**
 * Tesla Exact Typography Values
 * Based on Tesla's exact typography system
 */

export const teslaExactTypography = {
  // Font Sizes (exact in pixels)
  fontSize: {
    xs: '12px',      // 0.75rem
    sm: '14px',      // 0.875rem
    base: '16px',    // 1rem
    lg: '18px',      // 1.125rem
    xl: '20px',      // 1.25rem
    '2xl': '24px',   // 1.5rem
    '3xl': '30px',   // 1.875rem
    '4xl': '36px',   // 2.25rem
    '5xl': '48px',   // 3rem
    '6xl': '60px',   // 3.75rem
    '7xl': '72px',   // 4.5rem
    '8xl': '96px',   // 6rem
  },
  
  // Line Heights (exact)
  lineHeight: {
    none: 1,
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.75,
    loose: 2,
  },
  
  // Letter Spacing (exact)
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
  },
  
  // Font Weights (exact)
  fontWeight: {
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  
  // Typography Scale (exact combinations)
  scale: {
    hero: {
      fontSize: 'clamp(40px, 8vw, 96px)',
      lineHeight: 1.1,
      letterSpacing: '-0.04em',
      fontWeight: 700,
    },
    display: {
      fontSize: 'clamp(30px, 5vw, 60px)',
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
      fontWeight: 600,
    },
    h1: {
      fontSize: '36px',
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
      fontWeight: 600,
    },
    h2: {
      fontSize: '30px',
      lineHeight: 1.25,
      letterSpacing: '-0.02em',
      fontWeight: 600,
    },
    h3: {
      fontSize: '24px',
      lineHeight: 1.3,
      letterSpacing: '-0.02em',
      fontWeight: 600,
    },
    h4: {
      fontSize: '20px',
      lineHeight: 1.4,
      letterSpacing: '-0.01em',
      fontWeight: 600,
    },
    h5: {
      fontSize: '18px',
      lineHeight: 1.5,
      letterSpacing: '0',
      fontWeight: 600,
    },
    h6: {
      fontSize: '16px',
      lineHeight: 1.5,
      letterSpacing: '0',
      fontWeight: 600,
    },
    body: {
      fontSize: '16px',
      lineHeight: 1.5,
      letterSpacing: '0',
      fontWeight: 400,
    },
    small: {
      fontSize: '14px',
      lineHeight: 1.5,
      letterSpacing: '0',
      fontWeight: 400,
    },
    caption: {
      fontSize: '12px',
      lineHeight: 1.5,
      letterSpacing: '0.01em',
      fontWeight: 400,
    },
  },
} as const;

// Helper functions
export function getTeslaFontSize(size: keyof typeof teslaExactTypography.fontSize): string {
  return teslaExactTypography.fontSize[size];
}

export function getTeslaLineHeight(size: keyof typeof teslaExactTypography.lineHeight): number {
  return teslaExactTypography.lineHeight[size];
}

export function getTeslaLetterSpacing(size: keyof typeof teslaExactTypography.letterSpacing): string {
  return teslaExactTypography.letterSpacing[size];
}

export function getTeslaTypographyScale(scale: keyof typeof teslaExactTypography.scale) {
  return teslaExactTypography.scale[scale];
}



