/**
 * Tesla Exact Spacing Values
 * Based on Tesla's exact spacing system
 */

export const teslaExactSpacing = {
  // Padding Values (exact)
  padding: {
    xs: '8px',      // 0.5rem
    sm: '12px',     // 0.75rem
    md: '16px',     // 1rem
    lg: '24px',     // 1.5rem
    xl: '32px',     // 2rem
    '2xl': '48px',  // 3rem
    '3xl': '64px',  // 4rem
    '4xl': '96px',  // 6rem
  },
  
  // Margin Values (exact)
  margin: {
    xs: '8px',
    sm: '12px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    '2xl': '48px',
    '3xl': '64px',
    '4xl': '96px',
  },
  
  // Gap Values (exact)
  gap: {
    xs: '8px',
    sm: '12px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    '2xl': '48px',
  },
  
  // Component Specific Spacing
  card: {
    padding: '24px',      // p-6
    gap: '16px',         // gap-4
    borderRadius: '8px',  // rounded-lg
  },
  
  button: {
    paddingX: {
      sm: '16px',  // px-4
      md: '24px',  // px-6
      lg: '32px',  // px-8
    },
    paddingY: {
      sm: '8px',   // py-2
      md: '12px',  // py-3
      lg: '16px',  // py-4
    },
    borderRadius: '4px',  // rounded-md
    minHeight: '44px',
  },
  
  input: {
    paddingX: '16px',     // px-4
    paddingY: '12px',     // py-3
    borderRadius: '4px',  // rounded-md
    borderWidth: '1px',
  },
  
  section: {
    paddingY: {
      mobile: '64px',   // py-16
      desktop: '128px', // py-32
    },
    paddingX: {
      mobile: '16px',   // px-4
      tablet: '24px',   // px-6
      desktop: '32px', // px-8
    },
  },
} as const;

// Helper functions
export function getTeslaPadding(size: keyof typeof teslaExactSpacing.padding): string {
  return teslaExactSpacing.padding[size];
}

export function getTeslaMargin(size: keyof typeof teslaExactSpacing.margin): string {
  return teslaExactSpacing.margin[size];
}

export function getTeslaGap(size: keyof typeof teslaExactSpacing.gap): string {
  return teslaExactSpacing.gap[size];
}



