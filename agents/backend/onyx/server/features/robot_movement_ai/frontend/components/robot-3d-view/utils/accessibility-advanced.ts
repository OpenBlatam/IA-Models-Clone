/**
 * Advanced accessibility utilities
 * @module robot-3d-view/utils/accessibility-advanced
 */

/**
 * ARIA live region priority
 */
export type AriaLivePriority = 'polite' | 'assertive' | 'off';

/**
 * Accessibility Manager class
 */
export class AccessibilityManager {
  private liveRegions: Map<string, HTMLElement> = new Map();
  private focusHistory: HTMLElement[] = [];
  private maxFocusHistory = 10;

  /**
   * Creates or gets a live region
   */
  getLiveRegion(id: string, priority: AriaLivePriority = 'polite'): HTMLElement {
    let region = this.liveRegions.get(id);

    if (!region) {
      region = document.createElement('div');
      region.id = id;
      region.setAttribute('role', 'status');
      region.setAttribute('aria-live', priority);
      region.setAttribute('aria-atomic', 'true');
      region.className = 'sr-only';
      document.body.appendChild(region);
      this.liveRegions.set(id, region);
    } else {
      region.setAttribute('aria-live', priority);
    }

    return region;
  }

  /**
   * Announces a message to screen readers
   */
  announce(message: string, priority: AriaLivePriority = 'polite'): void {
    const region = this.getLiveRegion('aria-live-region', priority);
    region.textContent = message;

    // Clear after announcement
    setTimeout(() => {
      region.textContent = '';
    }, 1000);
  }

  /**
   * Traps focus within an element
   */
  trapFocus(element: HTMLElement): () => void {
    const focusableElements = this.getFocusableElements(element);
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    element.addEventListener('keydown', handleKeyDown);

    // Focus first element
    firstElement?.focus();

    // Return cleanup function
    return () => {
      element.removeEventListener('keydown', handleKeyDown);
    };
  }

  /**
   * Gets focusable elements within a container
   */
  getFocusableElements(container: HTMLElement): HTMLElement[] {
    const selector = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');

    return Array.from(container.querySelectorAll<HTMLElement>(selector)).filter(
      (el) => {
        const style = window.getComputedStyle(el);
        return style.display !== 'none' && style.visibility !== 'hidden';
      }
    );
  }

  /**
   * Saves current focus
   */
  saveFocus(): void {
    const activeElement = document.activeElement as HTMLElement;
    if (activeElement && activeElement !== document.body) {
      this.focusHistory.push(activeElement);
      if (this.focusHistory.length > this.maxFocusHistory) {
        this.focusHistory.shift();
      }
    }
  }

  /**
   * Restores previous focus
   */
  restoreFocus(): void {
    const previousFocus = this.focusHistory.pop();
    if (previousFocus && document.contains(previousFocus)) {
      previousFocus.focus();
    }
  }

  /**
   * Moves focus to element
   */
  moveFocus(element: HTMLElement): void {
    this.saveFocus();
    element.focus();
  }

  /**
   * Checks if element is visible to screen readers
   */
  isVisibleToScreenReader(element: HTMLElement): boolean {
    const style = window.getComputedStyle(element);
    const ariaHidden = element.getAttribute('aria-hidden');
    const role = element.getAttribute('role');

    if (ariaHidden === 'true') return false;
    if (style.display === 'none') return false;
    if (style.visibility === 'hidden') return false;
    if (style.opacity === '0' && !role) return false;

    return true;
  }

  /**
   * Gets accessible name of element
   */
  getAccessibleName(element: HTMLElement): string {
    const ariaLabel = element.getAttribute('aria-label');
    if (ariaLabel) return ariaLabel;

    const ariaLabelledBy = element.getAttribute('aria-labelledby');
    if (ariaLabelledBy) {
      const labelElement = document.getElementById(ariaLabelledBy);
      if (labelElement) return labelElement.textContent || '';
    }

    const label = element.closest('label');
    if (label) return label.textContent || '';

    return element.textContent || '';
  }

  /**
   * Cleans up all live regions
   */
  cleanup(): void {
    this.liveRegions.forEach((region) => {
      document.body.removeChild(region);
    });
    this.liveRegions.clear();
    this.focusHistory = [];
  }
}

/**
 * Global accessibility manager instance
 */
export const accessibilityManager = new AccessibilityManager();



