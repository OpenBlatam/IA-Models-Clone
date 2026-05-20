/**
 * Animation utilities and constants
 */

export const animations = {
  fadeIn: 'animate-fade-in',
  slideUp: 'animate-slide-up',
  pulse: 'animate-pulse',
  spin: 'animate-spin',
}

export const transitions = {
  default: 'transition-all duration-200 ease-in-out',
  fast: 'transition-all duration-150 ease-in-out',
  slow: 'transition-all duration-300 ease-in-out',
  colors: 'transition-colors duration-200 ease-in-out',
  transform: 'transition-transform duration-200 ease-in-out',
}

export const hoverEffects = {
  lift: 'hover:scale-105 hover:shadow-lg transition-transform duration-200',
  glow: 'hover:shadow-xl hover:shadow-primary-500/20 transition-shadow duration-200',
  border: 'hover:border-primary-500 transition-colors duration-200',
}




