/**
 * Shared animation configurations for Framer Motion
 * 
 * Provides consistent animation timing and easing across the module
 */

export const ANIMATION_DURATION = {
  FAST: 0.15,
  NORMAL: 0.2,
  SLOW: 0.3,
} as const;

export const ANIMATION_EASING = {
  EASE_OUT: [0.16, 1, 0.3, 1],
  EASE_IN: [0.4, 0, 1, 1],
  EASE_IN_OUT: [0.4, 0, 0.2, 1],
} as const;

/**
 * Common animation variants for cards
 */
export const cardAnimations = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
  transition: { duration: ANIMATION_DURATION.NORMAL },
  whileHover: { scale: 1.02 },
} as const;

/**
 * Common animation variants for modals
 */
export const modalAnimations = {
  overlay: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
    transition: { duration: ANIMATION_DURATION.NORMAL },
  },
  content: {
    initial: { opacity: 0, scale: 0.95, y: -20 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exit: { opacity: 0, scale: 0.95, y: -20 },
    transition: { duration: ANIMATION_DURATION.NORMAL },
  },
} as const;

/**
 * Common animation variants for fade in
 */
export const fadeInAnimation = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: ANIMATION_DURATION.NORMAL },
} as const;



