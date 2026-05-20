/**
 * Tesla Spacing Utilities
 * Utility functions to get exact Tesla spacing values
 */

import { teslaExactSpacing } from './tesla-exact-spacing';

/**
 * Get exact padding value
 */
export function p(size: keyof typeof teslaExactSpacing.padding): string {
  return teslaExactSpacing.padding[size];
}

/**
 * Get exact margin value
 */
export function m(size: keyof typeof teslaExactSpacing.margin): string {
  return teslaExactSpacing.margin[size];
}

/**
 * Get exact gap value
 */
export function g(size: keyof typeof teslaExactSpacing.gap): string {
  return teslaExactSpacing.gap[size];
}

/**
 * Spacing map for Tailwind classes
 * Maps standard Tailwind spacing to Tesla exact values
 */
export const teslaSpacingMap = {
  // Padding
  'p-0': '0px',
  'p-1': teslaExactSpacing.padding.xs, // 8px
  'p-2': teslaExactSpacing.padding.sm, // 12px
  'p-3': teslaExactSpacing.padding.md, // 16px
  'p-4': teslaExactSpacing.padding.md, // 16px
  'p-5': teslaExactSpacing.padding.sm, // 12px (no exact match, using closest)
  'p-6': teslaExactSpacing.padding.lg, // 24px
  'p-8': teslaExactSpacing.padding.xl, // 32px
  
  // Padding X
  'px-1': teslaExactSpacing.padding.xs,
  'px-2': teslaExactSpacing.padding.sm,
  'px-3': teslaExactSpacing.padding.md,
  'px-4': teslaExactSpacing.padding.md,
  'px-6': teslaExactSpacing.padding.lg,
  'px-8': teslaExactSpacing.padding.xl,
  
  // Padding Y
  'py-1': teslaExactSpacing.padding.xs,
  'py-2': teslaExactSpacing.padding.sm,
  'py-3': teslaExactSpacing.padding.md,
  'py-4': teslaExactSpacing.padding.md,
  'py-6': teslaExactSpacing.padding.lg,
  'py-8': teslaExactSpacing.padding.xl,
  'py-12': teslaExactSpacing.padding['2xl'], // 48px
  'py-16': teslaExactSpacing.padding['3xl'], // 64px
  
  // Margin
  'm-0': '0px',
  'm-1': teslaExactSpacing.margin.xs,
  'm-2': teslaExactSpacing.margin.sm,
  'm-3': teslaExactSpacing.margin.md,
  'm-4': teslaExactSpacing.margin.md,
  'm-6': teslaExactSpacing.margin.lg,
  'm-8': teslaExactSpacing.margin.xl,
  
  // Margin X
  'mx-1': teslaExactSpacing.margin.xs,
  'mx-2': teslaExactSpacing.margin.sm,
  'mx-3': teslaExactSpacing.margin.md,
  'mx-4': teslaExactSpacing.margin.md,
  'mx-6': teslaExactSpacing.margin.lg,
  'mx-8': teslaExactSpacing.margin.xl,
  
  // Margin Y
  'my-1': teslaExactSpacing.margin.xs,
  'my-2': teslaExactSpacing.margin.sm,
  'my-3': teslaExactSpacing.margin.md,
  'my-4': teslaExactSpacing.margin.md,
  'my-6': teslaExactSpacing.margin.lg,
  'my-8': teslaExactSpacing.margin.xl,
  
  // Margin Top
  'mt-1': teslaExactSpacing.margin.xs,
  'mt-2': teslaExactSpacing.margin.sm,
  'mt-3': teslaExactSpacing.margin.md,
  'mt-4': teslaExactSpacing.margin.md,
  'mt-6': teslaExactSpacing.margin.lg,
  'mt-8': teslaExactSpacing.margin.xl,
  
  // Margin Bottom
  'mb-1': teslaExactSpacing.margin.xs,
  'mb-2': teslaExactSpacing.margin.sm,
  'mb-3': teslaExactSpacing.margin.md,
  'mb-4': teslaExactSpacing.margin.md,
  'mb-6': teslaExactSpacing.margin.lg,
  'mb-8': teslaExactSpacing.margin.xl,
  
  // Gap
  'gap-1': teslaExactSpacing.gap.xs,
  'gap-2': teslaExactSpacing.gap.sm,
  'gap-3': teslaExactSpacing.gap.md,
  'gap-4': teslaExactSpacing.gap.md,
  'gap-6': teslaExactSpacing.gap.lg,
  'gap-8': teslaExactSpacing.gap.xl,
} as const;

/**
 * Get Tesla spacing value from Tailwind class
 */
export function getTeslaSpacingFromClass(className: keyof typeof teslaSpacingMap): string {
  return teslaSpacingMap[className];
}



