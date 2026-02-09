/**
 * Screen reader utilities
 * @module robot-3d-view/utils/screen-reader
 */

import { accessibilityManager } from './accessibility-advanced';

/**
 * Screen Reader Manager class
 */
export class ScreenReaderManager {
  /**
   * Announces text to screen readers
   */
  announce(text: string, priority: 'polite' | 'assertive' = 'polite'): void {
    accessibilityManager.announce(text, priority);
  }

  /**
   * Announces page title
   */
  announcePageTitle(title: string): void {
    document.title = title;
    this.announce(`Page: ${title}`, 'assertive');
  }

  /**
   * Announces status change
   */
  announceStatus(status: string): void {
    this.announce(`Status: ${status}`, 'polite');
  }

  /**
   * Announces error
   */
  announceError(message: string): void {
    this.announce(`Error: ${message}`, 'assertive');
  }

  /**
   * Announces success
   */
  announceSuccess(message: string): void {
    this.announce(`Success: ${message}`, 'polite');
  }

  /**
   * Announces navigation
   */
  announceNavigation(target: string): void {
    this.announce(`Navigated to ${target}`, 'polite');
  }

  /**
   * Announces form field label
   */
  announceFormField(label: string, value?: string): void {
    const message = value ? `${label}: ${value}` : label;
    this.announce(message, 'polite');
  }

  /**
   * Announces list item
   */
  announceListItem(index: number, total: number, label: string): void {
    this.announce(`${label}, item ${index + 1} of ${total}`, 'polite');
  }

  /**
   * Announces loading state
   */
  announceLoading(message: string): void {
    this.announce(`Loading: ${message}`, 'polite');
  }

  /**
   * Announces completion
   */
  announceComplete(message: string): void {
    this.announce(`Complete: ${message}`, 'polite');
  }

  /**
   * Checks if screen reader is likely active
   */
  isScreenReaderActive(): boolean {
    // Check for common screen reader indicators
    const hasAriaLive = document.querySelector('[aria-live]');
    const hasScreenReaderClass = document.querySelector('.sr-only');
    const hasHighContrast = window.matchMedia('(prefers-contrast: high)').matches;
    const hasReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    return !!(hasAriaLive || hasScreenReaderClass || hasHighContrast || hasReducedMotion);
  }

  /**
   * Gets screen reader friendly text
   */
  getScreenReaderText(element: HTMLElement): string {
    const ariaLabel = element.getAttribute('aria-label');
    if (ariaLabel) return ariaLabel;

    const ariaLabelledBy = element.getAttribute('aria-labelledby');
    if (ariaLabelledBy) {
      const labelElement = document.getElementById(ariaLabelledBy);
      if (labelElement) return labelElement.textContent || '';
    }

    const title = element.getAttribute('title');
    if (title) return title;

    return element.textContent || '';
  }
}

/**
 * Global screen reader manager instance
 */
export const screenReaderManager = new ScreenReaderManager();



