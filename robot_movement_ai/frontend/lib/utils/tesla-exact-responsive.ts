/**
 * Tesla Exact Responsive Values
 * Exact breakpoints and responsive spacing
 */

export const teslaExactResponsive = {
  // Breakpoints (exact)
  breakpoints: {
    xs: '475px',
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
    '3xl': '1920px',
  },

  // Responsive Font Sizes (exact)
  fontSize: {
    hero: {
      mobile: '40px',
      tablet: '60px',
      desktop: '96px',
      clamp: 'clamp(40px, 8vw, 96px)',
    },
    display: {
      mobile: '30px',
      tablet: '45px',
      desktop: '60px',
      clamp: 'clamp(30px, 5vw, 60px)',
    },
    h1: {
      mobile: '28px',
      tablet: '32px',
      desktop: '36px',
    },
    h2: {
      mobile: '24px',
      tablet: '28px',
      desktop: '30px',
    },
    h3: {
      mobile: '20px',
      tablet: '22px',
      desktop: '24px',
    },
    body: {
      mobile: '14px',
      tablet: '15px',
      desktop: '16px',
    },
  },

  // Responsive Spacing (exact)
  spacing: {
    section: {
      mobile: '64px',
      tablet: '96px',
      desktop: '128px',
    },
    container: {
      mobile: '16px',
      tablet: '24px',
      desktop: '32px',
    },
    card: {
      mobile: '16px',
      tablet: '20px',
      desktop: '24px',
    },
    button: {
      mobile: '12px 24px',
      tablet: '12px 24px',
      desktop: '16px 32px',
    },
  },

  // Responsive Grid Columns (exact)
  gridColumns: {
    mobile: 1,
    tablet: 2,
    desktop: 3,
    wide: 4,
  },

  // Responsive Gaps (exact)
  gaps: {
    mobile: '16px',
    tablet: '24px',
    desktop: '32px',
  },
} as const;

// Helper functions
export function getTeslaBreakpoint(size: keyof typeof teslaExactResponsive.breakpoints): string {
  return teslaExactResponsive.breakpoints[size];
}

export function getTeslaResponsiveFontSize(
  element: keyof typeof teslaExactResponsive.fontSize,
  breakpoint: 'mobile' | 'tablet' | 'desktop' | 'clamp'
): string {
  return teslaExactResponsive.fontSize[element][breakpoint];
}

export function getTeslaResponsiveSpacing(
  element: keyof typeof teslaExactResponsive.spacing,
  breakpoint: 'mobile' | 'tablet' | 'desktop'
): string {
  return teslaExactResponsive.spacing[element][breakpoint];
}



