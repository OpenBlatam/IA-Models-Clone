/**
 * Accessibility utilities.
 * Provides utilities for improving accessibility in the application.
 */

/**
 * ARIA live region priorities.
 */
export enum AriaLivePriority {
  POLITE = 'polite',
  ASSERTIVE = 'assertive',
  OFF = 'off',
}

/**
 * Options for creating an ARIA live region.
 */
export interface AriaLiveRegionOptions {
  /**
   * Priority of the live region.
   */
  priority?: AriaLivePriority;
  /**
   * Whether the region is atomic.
   */
  atomic?: boolean;
  /**
   * Whether the region is relevant.
   */
  relevant?: 'additions' | 'removals' | 'text' | 'all';
}

/**
 * Creates an ARIA live region element.
 */
export function createAriaLiveRegion(
  id: string,
  options: AriaLiveRegionOptions = {}
): HTMLElement {
  const {
    priority = AriaLivePriority.POLITE,
    atomic = false,
    relevant = 'additions text',
  } = options;

  // Remove existing region if it exists
  const existing = document.getElementById(id);
  if (existing) {
    existing.remove();
  }

  const region = document.createElement('div');
  region.id = id;
  region.setAttribute('role', 'status');
  region.setAttribute('aria-live', priority);
  region.setAttribute('aria-atomic', String(atomic));
  region.setAttribute('aria-relevant', relevant);
  region.className = 'sr-only'; // Screen reader only
  region.style.cssText = `
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  `;

  document.body.appendChild(region);
  return region;
}

/**
 * Announces a message to screen readers.
 */
export function announceToScreenReader(
  message: string,
  priority: AriaLivePriority = AriaLivePriority.POLITE
): void {
  if (typeof window === 'undefined') {
    return;
  }

  const regionId = `aria-live-region-${priority}`;
  let region = document.getElementById(regionId);

  if (!region) {
    region = createAriaLiveRegion(regionId, { priority });
  }

  // Clear and set new message
  region.textContent = '';
  // Use setTimeout to ensure the message is announced
  setTimeout(() => {
    region.textContent = message;
  }, 100);
}

/**
 * Gets accessible label for an element.
 */
export function getAccessibleLabel(element: HTMLElement): string | null {
  // Check aria-label first
  const ariaLabel = element.getAttribute('aria-label');
  if (ariaLabel) {
    return ariaLabel;
  }

  // Check aria-labelledby
  const labelledBy = element.getAttribute('aria-labelledby');
  if (labelledBy) {
    const labelElement = document.getElementById(labelledBy);
    if (labelElement) {
      return labelElement.textContent || labelElement.getAttribute('aria-label');
    }
  }

  // Check for associated label
  const id = element.id;
  if (id) {
    const label = document.querySelector(`label[for="${id}"]`);
    if (label) {
      return label.textContent;
    }
  }

  // Check for title attribute
  const title = element.getAttribute('title');
  if (title) {
    return title;
  }

  // Check for text content (for buttons, links, etc.)
  const textContent = element.textContent?.trim();
  if (textContent) {
    return textContent;
  }

  return null;
}

/**
 * Checks if an element is focusable.
 */
export function isFocusable(element: HTMLElement): boolean {
  // Check if element has tabindex
  const tabIndex = element.getAttribute('tabindex');
  if (tabIndex !== null) {
    const index = parseInt(tabIndex, 10);
    return index >= 0 || (index === -1 && element.hasAttribute('contenteditable'));
  }

  // Check native focusable elements
  const focusableSelectors = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[contenteditable="true"]',
    '[tabindex]:not([tabindex="-1"])',
  ];

  return focusableSelectors.some((selector) => element.matches(selector));
}

/**
 * Gets all focusable elements within a container.
 */
export function getFocusableElements(
  container: HTMLElement | Document = document
): HTMLElement[] {
  const focusableSelectors = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[contenteditable="true"]',
    '[tabindex]:not([tabindex="-1"])',
  ].join(', ');

  const elements = Array.from(
    container.querySelectorAll<HTMLElement>(focusableSelectors)
  );

  return elements.filter((el) => {
    // Filter out hidden elements
    const style = window.getComputedStyle(el);
    return (
      style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      style.opacity !== '0' &&
      !el.hasAttribute('aria-hidden')
    );
  });
}

/**
 * Traps focus within a container element.
 */
export function trapFocus(container: HTMLElement): () => void {
  const focusableElements = getFocusableElements(container);
  const firstElement = focusableElements[0];
  const lastElement = focusableElements[focusableElements.length - 1];

  const handleKeyDown = (e: KeyboardEvent): void => {
    if (e.key !== 'Tab') {
      return;
    }

    if (e.shiftKey) {
      // Shift + Tab
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      }
    } else {
      // Tab
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    }
  };

  container.addEventListener('keydown', handleKeyDown);

  // Focus first element
  firstElement?.focus();

  // Return cleanup function
  return () => {
    container.removeEventListener('keydown', handleKeyDown);
  };
}

/**
 * Restores focus to a previously focused element.
 */
export function restoreFocus(element: HTMLElement | null): void {
  if (element && typeof element.focus === 'function') {
    element.focus();
  }
}

/**
 * Saves the currently focused element.
 */
export function saveFocus(): HTMLElement | null {
  return document.activeElement as HTMLElement | null;
}

/**
 * Checks if reduced motion is preferred.
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Checks if high contrast mode is enabled.
 */
export function prefersHighContrast(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return (
    window.matchMedia('(prefers-contrast: high)').matches ||
    window.matchMedia('(prefers-contrast: more)').matches
  );
}

/**
 * Gets color contrast ratio between two colors.
 * Returns a value between 1 and 21.
 */
export function getContrastRatio(color1: string, color2: string): number {
  const getLuminance = (color: string): number => {
    // Convert hex to RGB
    const hex = color.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;

    // Apply gamma correction
    const [rLinear, gLinear, bLinear] = [r, g, b].map((val) => {
      return val <= 0.03928 ? val / 12.92 : Math.pow((val + 0.055) / 1.055, 2.4);
    });

    // Calculate relative luminance
    return 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear;
  };

  const luminance1 = getLuminance(color1);
  const luminance2 = getLuminance(color2);

  const lighter = Math.max(luminance1, luminance2);
  const darker = Math.min(luminance1, luminance2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Checks if contrast ratio meets WCAG AA standards.
 * Returns true if ratio is at least 4.5:1 for normal text or 3:1 for large text.
 */
export function meetsContrastRatio(
  color1: string,
  color2: string,
  isLargeText = false
): boolean {
  const ratio = getContrastRatio(color1, color2);
  const requiredRatio = isLargeText ? 3 : 4.5;
  return ratio >= requiredRatio;
}

/**
 * Validates ARIA attributes on an element.
 */
export function validateAriaAttributes(element: HTMLElement): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // Check for aria-label or aria-labelledby
  const hasLabel =
    element.hasAttribute('aria-label') || element.hasAttribute('aria-labelledby');
  const role = element.getAttribute('role');

  // Interactive elements should have labels
  if (
    ['button', 'link', 'menuitem', 'tab', 'option'].includes(role || '') &&
    !hasLabel &&
    !getAccessibleLabel(element)
  ) {
    errors.push('Interactive element with role should have aria-label or aria-labelledby');
  }

  // Check for aria-describedby references
  const describedBy = element.getAttribute('aria-describedby');
  if (describedBy) {
    const ids = describedBy.split(' ');
    ids.forEach((id) => {
      if (!document.getElementById(id)) {
        errors.push(`aria-describedby references non-existent id: ${id}`);
      }
    });
  }

  // Check for aria-labelledby references
  const labelledBy = element.getAttribute('aria-labelledby');
  if (labelledBy) {
    const ids = labelledBy.split(' ');
    ids.forEach((id) => {
      if (!document.getElementById(id)) {
        errors.push(`aria-labelledby references non-existent id: ${id}`);
      }
    });
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}




