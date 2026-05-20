/**
 * DOM utility functions.
 * Provides helper functions for DOM manipulation and queries.
 */

/**
 * Gets an element by selector.
 * @param selector - CSS selector
 * @param parent - Parent element (default: document)
 * @returns Element or null
 */
export function querySelector<T extends Element = Element>(
  selector: string,
  parent: Document | Element = document
): T | null {
  return parent.querySelector<T>(selector);
}

/**
 * Gets all elements by selector.
 * @param selector - CSS selector
 * @param parent - Parent element (default: document)
 * @returns NodeList of elements
 */
export function querySelectorAll<T extends Element = Element>(
  selector: string,
  parent: Document | Element = document
): NodeListOf<T> {
  return parent.querySelectorAll<T>(selector);
}

/**
 * Checks if an element matches a selector.
 * @param element - Element to check
 * @param selector - CSS selector
 * @returns True if element matches selector
 */
export function matches(element: Element, selector: string): boolean {
  return element.matches(selector);
}

/**
 * Gets the closest ancestor matching a selector.
 * @param element - Starting element
 * @param selector - CSS selector
 * @returns Closest matching element or null
 */
export function closest<T extends Element = Element>(
  element: Element,
  selector: string
): T | null {
  return element.closest<T>(selector);
}

/**
 * Scrolls an element into view.
 * @param element - Element to scroll
 * @param options - Scroll options
 */
export function scrollIntoView(
  element: Element,
  options: ScrollIntoViewOptions = {}
): void {
  element.scrollIntoView({
    behavior: 'smooth',
    block: 'nearest',
    inline: 'nearest',
    ...options,
  });
}

/**
 * Gets the bounding rectangle of an element.
 * @param element - Element
 * @returns Bounding rectangle
 */
export function getBoundingClientRect(element: Element): DOMRect {
  return element.getBoundingClientRect();
}

/**
 * Checks if an element is visible in viewport.
 * @param element - Element to check
 * @param threshold - Visibility threshold (0-1)
 * @returns True if element is visible
 */
export function isElementVisible(
  element: Element,
  threshold: number = 0
): boolean {
  const rect = element.getBoundingClientRect();
  const windowHeight =
    window.innerHeight || document.documentElement.clientHeight;
  const windowWidth = window.innerWidth || document.documentElement.clientWidth;

  const vertInView =
    rect.top <= windowHeight * (1 - threshold) &&
    rect.bottom >= windowHeight * threshold;
  const horInView =
    rect.left <= windowWidth * (1 - threshold) &&
    rect.right >= windowWidth * threshold;

  return vertInView && horInView;
}

/**
 * Gets computed styles of an element.
 * @param element - Element
 * @returns Computed styles
 */
export function getComputedStyles(element: Element): CSSStyleDeclaration {
  return window.getComputedStyle(element);
}

/**
 * Adds a class to an element.
 * @param element - Element
 * @param className - Class name to add
 */
export function addClass(element: Element, className: string): void {
  element.classList.add(className);
}

/**
 * Removes a class from an element.
 * @param element - Element
 * @param className - Class name to remove
 */
export function removeClass(element: Element, className: string): void {
  element.classList.remove(className);
}

/**
 * Toggles a class on an element.
 * @param element - Element
 * @param className - Class name to toggle
 * @returns True if class was added
 */
export function toggleClass(
  element: Element,
  className: string
): boolean {
  return element.classList.toggle(className);
}

/**
 * Checks if an element has a class.
 * @param element - Element
 * @param className - Class name to check
 * @returns True if element has class
 */
export function hasClass(element: Element, className: string): boolean {
  return element.classList.contains(className);
}

