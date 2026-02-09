/**
 * Utility for merging Tailwind CSS classes
 * Uses clsx and tailwind-merge for optimal class handling
 */
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]): string => {
  return twMerge(clsx(inputs));
};







