/**
 * Class name helper utilities
 */

import { cn } from './cn';

/**
 * Create conditional class names
 */
export function classNames(
  ...classes: Array<string | Record<string, boolean> | undefined | null | false>
): string {
  return cn(...classes);
}

/**
 * Create variant-based class names
 */
export function createVariants<T extends string>(
  base: string,
  variants: Record<T, string>
) {
  return (variant: T): string => {
    return cn(base, variants[variant]);
  };
}

/**
 * Create size-based class names
 */
export function createSizes(
  base: string,
  sizes: Record<string, string>
) {
  return (size: string): string => {
    return cn(base, sizes[size]);
  };
}

/**
 * Combine variant and size classes
 */
export function createComponentClasses(
  base: string,
  variants: Record<string, string>,
  sizes: Record<string, string>
) {
  return (variant: string, size: string): string => {
    return cn(base, variants[variant], sizes[size]);
  };
}

/**
 * Create responsive classes
 */
export function responsive(
  base: string,
  sm?: string,
  md?: string,
  lg?: string,
  xl?: string
): string {
  return cn(
    base,
    sm && `sm:${sm}`,
    md && `md:${md}`,
    lg && `lg:${lg}`,
    xl && `xl:${xl}`
  );
}



