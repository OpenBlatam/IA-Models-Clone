/**
 * Tesla Exact Interactive States
 * Exact values for hover, focus, active, and disabled states
 */

export const teslaExactInteractiveStates = {
  // Button States (exact)
  button: {
    primary: {
      default: {
        backgroundColor: '#0062cc',
        color: '#ffffff',
        border: 'none',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
      },
      hover: {
        backgroundColor: '#0052a3',
        color: '#ffffff',
        boxShadow: '0 4px 6px -1px rgba(0, 98, 204, 0.2)',
        transform: 'scale(1.02)',
      },
      active: {
        backgroundColor: '#004280',
        transform: 'scale(0.98)',
      },
      focus: {
        outline: '2px solid #0062cc',
        outlineOffset: '2px',
        ringWidth: '2px',
        ringColor: '#0062cc',
        ringOffset: '2px',
      },
      disabled: {
        backgroundColor: '#e5e7eb',
        color: '#9ca3af',
        opacity: 0.5,
        cursor: 'not-allowed',
      },
    },
    secondary: {
      default: {
        backgroundColor: '#ffffff',
        color: '#171a20',
        border: '1px solid #d1d5db',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
      },
      hover: {
        backgroundColor: '#f9fafb',
        borderColor: '#9ca3af',
        boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
        transform: 'scale(1.02)',
      },
      active: {
        backgroundColor: '#f3f4f6',
        transform: 'scale(0.98)',
      },
      focus: {
        outline: '2px solid #0062cc',
        outlineOffset: '2px',
        ringWidth: '2px',
        ringColor: '#0062cc',
        ringOffset: '2px',
      },
      disabled: {
        backgroundColor: '#ffffff',
        color: '#9ca3af',
        borderColor: '#e5e7eb',
        opacity: 0.5,
        cursor: 'not-allowed',
      },
    },
    tertiary: {
      default: {
        backgroundColor: 'transparent',
        color: '#393c41',
        border: 'none',
      },
      hover: {
        backgroundColor: '#f9fafb',
        color: '#171a20',
      },
      active: {
        backgroundColor: '#f3f4f6',
      },
      focus: {
        outline: '2px solid #0062cc',
        outlineOffset: '2px',
      },
      disabled: {
        color: '#9ca3af',
        opacity: 0.5,
        cursor: 'not-allowed',
      },
    },
  },

  // Input States (exact)
  input: {
    default: {
      backgroundColor: '#ffffff',
      border: '1px solid #d1d5db',
      color: '#171a20',
      boxShadow: 'none',
    },
    hover: {
      borderColor: '#9ca3af',
    },
    focus: {
      borderColor: '#0062cc',
      boxShadow: '0 0 0 3px rgba(0, 98, 204, 0.1)',
      outline: 'none',
    },
    error: {
      borderColor: '#ef4444',
      boxShadow: '0 0 0 3px rgba(239, 68, 68, 0.1)',
    },
    disabled: {
      backgroundColor: '#f9fafb',
      borderColor: '#e5e7eb',
      color: '#9ca3af',
      cursor: 'not-allowed',
    },
  },

  // Card States (exact)
  card: {
    default: {
      backgroundColor: '#ffffff',
      border: '1px solid #e5e7eb',
      boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    },
    hover: {
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      transform: 'translateY(-2px)',
      borderColor: '#d1d5db',
    },
    active: {
      boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
      transform: 'translateY(0)',
    },
  },

  // Link States (exact)
  link: {
    default: {
      color: '#0062cc',
      textDecoration: 'none',
    },
    hover: {
      color: '#0052a3',
      textDecoration: 'underline',
      textUnderlineOffset: '2px',
    },
    visited: {
      color: '#0052a3',
    },
    focus: {
      outline: '2px solid #0062cc',
      outlineOffset: '2px',
      borderRadius: '2px',
    },
  },

  // Checkbox/Radio States (exact)
  checkbox: {
    default: {
      border: '1px solid #b5b5b5',
      backgroundColor: '#ffffff',
      width: '18px',
      height: '18px',
    },
    checked: {
      borderColor: '#0062cc',
      backgroundColor: '#0062cc',
    },
    hover: {
      borderColor: '#0062cc',
    },
    focus: {
      outline: '2px solid #0062cc',
      outlineOffset: '2px',
    },
    disabled: {
      borderColor: '#e5e7eb',
      backgroundColor: '#f9fafb',
      opacity: 0.5,
      cursor: 'not-allowed',
    },
  },

  // Switch States (exact)
  switch: {
    default: {
      backgroundColor: '#e5e7eb',
      width: '44px',
      height: '24px',
    },
    checked: {
      backgroundColor: '#0062cc',
    },
    thumb: {
      width: '20px',
      height: '20px',
      backgroundColor: '#ffffff',
      boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.2)',
    },
    disabled: {
      backgroundColor: '#f3f4f6',
      opacity: 0.5,
      cursor: 'not-allowed',
    },
  },

  // Tab States (exact)
  tab: {
    default: {
      color: '#393c41',
      borderBottom: '2px solid transparent',
      backgroundColor: 'transparent',
    },
    active: {
      color: '#0062cc',
      borderBottomColor: '#0062cc',
      backgroundColor: 'rgba(0, 98, 204, 0.05)',
    },
    hover: {
      color: '#171a20',
      borderBottomColor: '#d1d5db',
      backgroundColor: '#f9fafb',
    },
  },

  // Badge States (exact)
  badge: {
    default: {
      backgroundColor: '#e5e7eb',
      color: '#171a20',
      padding: '4px 12px',
      borderRadius: '9999px',
      fontSize: '12px',
      fontWeight: 500,
    },
    primary: {
      backgroundColor: '#0062cc',
      color: '#ffffff',
    },
    success: {
      backgroundColor: '#10b981',
      color: '#ffffff',
    },
    error: {
      backgroundColor: '#ef4444',
      color: '#ffffff',
    },
    warning: {
      backgroundColor: '#f59e0b',
      color: '#ffffff',
    },
  },
} as const;

// Helper functions
export function getTeslaButtonState(
  variant: 'primary' | 'secondary' | 'tertiary',
  state: 'default' | 'hover' | 'active' | 'focus' | 'disabled'
) {
  return teslaExactInteractiveStates.button[variant][state];
}

export function getTeslaInputState(state: 'default' | 'hover' | 'focus' | 'error' | 'disabled') {
  return teslaExactInteractiveStates.input[state];
}

export function getTeslaCardState(state: 'default' | 'hover' | 'active') {
  return teslaExactInteractiveStates.card[state];
}

export function getTeslaLinkState(state: 'default' | 'hover' | 'visited' | 'focus') {
  return teslaExactInteractiveStates.link[state];
}

export function getTeslaCheckboxState(state: 'default' | 'checked' | 'hover' | 'focus' | 'disabled') {
  return teslaExactInteractiveStates.checkbox[state];
}

export function getTeslaTabState(state: 'default' | 'active' | 'hover') {
  return teslaExactInteractiveStates.tab[state];
}

export function getTeslaBadgeState(variant: 'default' | 'primary' | 'success' | 'error' | 'warning') {
  return teslaExactInteractiveStates.badge[variant];
}



