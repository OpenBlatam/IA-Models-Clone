/**
 * Accessibility utilities and constants
 */

export const ACCESSIBILITY_ROLES = {
  BUTTON: 'button',
  LINK: 'link',
  HEADER: 'header',
  TEXT: 'text',
  IMAGE: 'image',
  NONE: 'none',
} as const;

export const ACCESSIBILITY_STATES = {
  SELECTED: 'selected',
  DISABLED: 'disabled',
  CHECKED: 'checked',
  UNCHECKED: 'unchecked',
  BUSY: 'busy',
  EXPANDED: 'expanded',
  COLLAPSED: 'collapsed',
} as const;

/**
 * Generate accessibility label from text content
 */
export function generateAccessibilityLabel(text: string): string {
  return text.trim();
}

/**
 * Generate accessibility hint for actions
 */
export function generateAccessibilityHint(action: string, target?: string): string {
  if (target) {
    return `${action} ${target}`;
  }
  return action;
}

/**
 * Check if text should be read as heading
 */
export function isHeading(text: string, level?: number): boolean {
  // Simple heuristic: if text is short and uppercase, likely a heading
  if (text.length < 50 && text === text.toUpperCase()) {
    return true;
  }
  return false;
}

/**
 * Format number for screen readers
 */
export function formatNumberForScreenReader(num: number): string {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(1)} million`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)} thousand`;
  }
  return num.toString();
}

/**
 * Format date for screen readers
 */
export function formatDateForScreenReader(date: Date): string {
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}


