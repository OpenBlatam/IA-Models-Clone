/**
 * Tooltip utilities
 * @module robot-3d-view/utils/tooltips
 */

/**
 * Tooltip configuration
 */
export interface TooltipConfig {
  content: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  disabled?: boolean;
}

/**
 * Default tooltip configuration
 */
export const DEFAULT_TOOLTIP_CONFIG: Required<TooltipConfig> = {
  content: '',
  position: 'top',
  delay: 300,
  disabled: false,
};

/**
 * Creates tooltip configuration
 * 
 * @param content - Tooltip content
 * @param options - Additional options
 * @returns Tooltip configuration
 */
export function createTooltip(
  content: string,
  options?: Partial<TooltipConfig>
): TooltipConfig {
  return {
    ...DEFAULT_TOOLTIP_CONFIG,
    content,
    ...options,
  };
}

/**
 * Gets tooltip position classes
 * 
 * @param position - Tooltip position
 * @returns Tailwind classes for position
 */
export function getTooltipPositionClasses(position: TooltipConfig['position'] = 'top'): string {
  const classes = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };
  return classes[position];
}



