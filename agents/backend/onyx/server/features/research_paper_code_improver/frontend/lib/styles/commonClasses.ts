/**
 * Common CSS Classes - Reusable class combinations
 * 
 * This file contains commonly used class combinations to reduce duplication
 * and ensure consistency across components.
 */

export const commonClasses = {
  // Card styles
  card: {
    base: 'bg-white rounded-lg shadow-md p-6',
    padding: {
      sm: 'p-4',
      md: 'p-6',
      lg: 'p-8',
    },
    shadow: {
      sm: 'shadow-sm',
      md: 'shadow-md',
      lg: 'shadow-lg',
    },
  },

  // Button base styles
  button: {
    base: 'inline-flex items-center justify-center font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
    sizes: {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg',
      icon: 'p-2',
    },
  },

  // Input styles
  input: {
    base: 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed',
    error: 'border-red-500 focus:ring-red-500',
    success: 'border-green-500 focus:ring-green-500',
  },

  // Textarea styles
  textarea: {
    base: 'w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-vertical disabled:bg-gray-100 disabled:cursor-not-allowed',
  },

  // Badge styles
  badge: {
    base: 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
    sizes: {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-2.5 py-0.5 text-xs',
      lg: 'px-3 py-1 text-sm',
    },
  },

  // Container styles
  container: {
    base: 'mx-auto px-4 sm:px-6 lg:px-8',
    maxWidth: {
      sm: 'max-w-screen-sm',
      md: 'max-w-screen-md',
      lg: 'max-w-screen-lg',
      xl: 'max-w-screen-xl',
      '2xl': 'max-w-screen-2xl',
    },
  },

  // Flexbox utilities
  flex: {
    center: 'flex items-center justify-center',
    between: 'flex items-center justify-between',
    start: 'flex items-center justify-start',
    end: 'flex items-center justify-end',
    col: 'flex flex-col',
    row: 'flex flex-row',
  },

  // Grid utilities
  grid: {
    '1': 'grid grid-cols-1',
    '2': 'grid grid-cols-1 md:grid-cols-2',
    '3': 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    '4': 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  },

  // Spacing utilities
  spacing: {
    section: 'space-y-8',
    card: 'space-y-6',
    form: 'space-y-4',
    group: 'space-y-2',
  },

  // Text utilities
  text: {
    heading: {
      h1: 'text-3xl font-bold text-gray-900',
      h2: 'text-2xl font-semibold text-gray-900',
      h3: 'text-xl font-semibold text-gray-900',
      h4: 'text-lg font-semibold text-gray-900',
    },
    body: {
      base: 'text-base text-gray-700',
      sm: 'text-sm text-gray-600',
      xs: 'text-xs text-gray-500',
    },
  },

  // Border utilities
  border: {
    base: 'border border-gray-200',
    light: 'border border-gray-100',
    dark: 'border border-gray-300',
    rounded: {
      sm: 'rounded-sm',
      md: 'rounded-md',
      lg: 'rounded-lg',
      xl: 'rounded-xl',
      full: 'rounded-full',
    },
  },

  // Background utilities
  background: {
    white: 'bg-white',
    gray: {
      50: 'bg-gray-50',
      100: 'bg-gray-100',
    },
    primary: {
      50: 'bg-primary-50',
      100: 'bg-primary-100',
    },
  },

  // Transition utilities
  transition: {
    base: 'transition-colors duration-200',
    all: 'transition-all duration-200',
    fast: 'transition-colors duration-150',
    slow: 'transition-colors duration-300',
  },
} as const

/**
 * Combine common classes with custom classes
 */
export function combineClasses(
  ...classes: (string | undefined | false | null)[]
): string {
  return classes.filter(Boolean).join(' ')
}




