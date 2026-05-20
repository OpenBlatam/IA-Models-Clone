/**
 * Tesla Animation Utilities
 * Exact animation configurations and helpers
 */

export const teslaAnimations = {
  // Duration values (exact)
  duration: {
    instant: '0ms',
    fast: '150ms',
    base: '200ms',
    slow: '300ms',
    slower: '400ms',
    slowest: '600ms',
  },

  // Easing curves (exact)
  easing: {
    linear: 'linear',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    spring: 'cubic-bezier(0.16, 1, 0.3, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },

  // Spring configurations
  spring: {
    gentle: { tension: 200, friction: 25 },
    normal: { tension: 280, friction: 60 },
    stiff: { tension: 400, friction: 80 },
    wobbly: { tension: 180, friction: 12 },
  },

  // Framer Motion variants
  variants: {
    fadeIn: {
      hidden: { opacity: 0 },
      visible: { opacity: 1 },
    },
    slideUp: {
      hidden: { opacity: 0, y: 20 },
      visible: { opacity: 1, y: 0 },
    },
    slideDown: {
      hidden: { opacity: 0, y: -20 },
      visible: { opacity: 1, y: 0 },
    },
    slideLeft: {
      hidden: { opacity: 0, x: 20 },
      visible: { opacity: 1, x: 0 },
    },
    slideRight: {
      hidden: { opacity: 0, x: -20 },
      visible: { opacity: 1, x: 0 },
    },
    scale: {
      hidden: { opacity: 0, scale: 0.9 },
      visible: { opacity: 1, scale: 1 },
    },
    rotate: {
      hidden: { opacity: 0, rotate: -10 },
      visible: { opacity: 1, rotate: 0 },
    },
  },

  // Stagger configurations
  stagger: {
    fast: 0.05,
    normal: 0.1,
    slow: 0.2,
  },

  // Hover effects
  hover: {
    scale: {
      subtle: 1.01,
      normal: 1.02,
      prominent: 1.05,
    },
    translateY: {
      subtle: -2,
      normal: -4,
      prominent: -8,
    },
    shadow: {
      sm: '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
      md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    },
  },

  // Tap effects
  tap: {
    scale: {
      subtle: 0.98,
      normal: 0.95,
      heavy: 0.9,
    },
  },
} as const;

// Helper functions
export function getTeslaDuration(size: keyof typeof teslaAnimations.duration): string {
  return teslaAnimations.duration[size];
}

export function getTeslaEasing(type: keyof typeof teslaAnimations.easing): string {
  return teslaAnimations.easing[type];
}

export function getTeslaSpring(type: keyof typeof teslaAnimations.spring) {
  return teslaAnimations.spring[type];
}

export function getTeslaVariant(name: keyof typeof teslaAnimations.variants) {
  return teslaAnimations.variants[name];
}

export function getTeslaHoverScale(size: keyof typeof teslaAnimations.hover.scale): number {
  return teslaAnimations.hover.scale[size];
}

export function getTeslaHoverShadow(size: keyof typeof teslaAnimations.hover.shadow): string {
  return teslaAnimations.hover.shadow[size];
}

// Animation presets
export const animationPresets = {
  // Page transitions
  pageTransition: {
    initial: { opacity: 0, y: 20, scale: 0.98 },
    animate: { opacity: 1, y: 0, scale: 1 },
    exit: { opacity: 0, y: -20, scale: 1.02 },
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
  },

  // Card entrance
  cardEntrance: {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] },
  },

  // Button hover
  buttonHover: {
    scale: 1.02,
    transition: { duration: 0.2, ease: [0.4, 0, 0.2, 1] },
  },

  // Button tap
  buttonTap: {
    scale: 0.98,
    transition: { duration: 0.1, ease: [0.4, 0, 0.2, 1] },
  },

  // Modal entrance
  modalEntrance: {
    initial: { opacity: 0, scale: 0.95, y: 20 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exit: { opacity: 0, scale: 0.95, y: 20 },
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },

  // Toast notification
  toastEntrance: {
    initial: { opacity: 0, x: 300 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 300 },
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
} as const;



