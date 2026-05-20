/**
 * Tesla Exact Layout System
 * Exact spacing, grid, and layout values
 */

export const teslaExactLayout = {
  // Container Max Widths (exact)
  container: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
    full: '100%',
  },

  // Grid Gaps (exact)
  gridGap: {
    xs: '8px',
    sm: '12px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    '2xl': '48px',
    '3xl': '64px',
  },

  // Section Padding (exact)
  sectionPadding: {
    mobile: {
      top: '64px',
      bottom: '64px',
      horizontal: '16px',
    },
    tablet: {
      top: '96px',
      bottom: '96px',
      horizontal: '24px',
    },
    desktop: {
      top: '128px',
      bottom: '128px',
      horizontal: '32px',
    },
  },

  // Component Spacing (exact)
  componentSpacing: {
    // Card spacing
    card: {
      padding: '24px',
      gap: '16px',
      marginBottom: '24px',
    },
    // Form spacing
    form: {
      fieldGap: '16px',
      labelMarginBottom: '8px',
      errorMarginTop: '4px',
      groupGap: '24px',
    },
    // Button group spacing
    buttonGroup: {
      gap: '12px',
      marginTop: '24px',
    },
    // List spacing
    list: {
      itemGap: '12px',
      itemPadding: '12px',
      sectionGap: '24px',
    },
    // Grid spacing
    grid: {
      gap: '24px',
      itemPadding: '24px',
    },
  },

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

  // Z-Index Scale (exact)
  zIndex: {
    base: 0,
    dropdown: 1000,
    sticky: 1020,
    fixed: 1030,
    modalBackdrop: 1040,
    modal: 1050,
    popover: 1060,
    tooltip: 1070,
    notification: 1080,
  },

  // Aspect Ratios (exact)
  aspectRatio: {
    square: '1 / 1',
    video: '16 / 9',
    photo: '4 / 3',
    wide: '21 / 9',
    portrait: '3 / 4',
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
} as const;

// Helper functions
export function getTeslaContainerWidth(size: keyof typeof teslaExactLayout.container): string {
  return teslaExactLayout.container[size];
}

export function getTeslaGridGap(size: keyof typeof teslaExactLayout.gridGap): string {
  return teslaExactLayout.gridGap[size];
}

export function getTeslaSectionPadding(breakpoint: 'mobile' | 'tablet' | 'desktop') {
  return teslaExactLayout.sectionPadding[breakpoint];
}

export function getTeslaComponentSpacing(component: keyof typeof teslaExactLayout.componentSpacing) {
  return teslaExactLayout.componentSpacing[component];
}

export function getTeslaZIndex(level: keyof typeof teslaExactLayout.zIndex): number {
  return teslaExactLayout.zIndex[level];
}

export function getTeslaAspectRatio(ratio: keyof typeof teslaExactLayout.aspectRatio): string {
  return teslaExactLayout.aspectRatio[ratio];
}



