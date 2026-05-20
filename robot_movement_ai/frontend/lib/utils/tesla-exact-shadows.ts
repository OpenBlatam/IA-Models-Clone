/**
 * Tesla Exact Shadow Values
 * Based on Tesla's exact shadow system
 */

export const teslaExactShadows = {
  // Box Shadows (exact)
  boxShadow: {
    xs: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    sm: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
  },
  
  // Text Shadows (exact)
  textShadow: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.1)',
    md: '0 2px 4px rgba(0, 0, 0, 0.15)',
    lg: '0 4px 8px rgba(0, 0, 0, 0.2)',
  },
  
  // Component Specific Shadows
  card: {
    default: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    hover: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    active: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  },
  
  button: {
    default: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    hover: '0 4px 6px -1px rgba(0, 98, 204, 0.2)',
    active: '0 1px 3px 0 rgba(0, 98, 204, 0.1)',
  },
  
  modal: {
    backdrop: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    content: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  },
  
  dropdown: {
    default: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  },
} as const;

// Helper functions
export function getTeslaBoxShadow(size: keyof typeof teslaExactShadows.boxShadow): string {
  return teslaExactShadows.boxShadow[size];
}

export function getTeslaTextShadow(size: keyof typeof teslaExactShadows.textShadow): string {
  return teslaExactShadows.textShadow[size];
}

export function getTeslaCardShadow(state: 'default' | 'hover' | 'active'): string {
  return teslaExactShadows.card[state];
}

export function getTeslaButtonShadow(state: 'default' | 'hover' | 'active'): string {
  return teslaExactShadows.button[state];
}



