/**
 * DOM utilities
 */

// Get element by ID
export function getElementById<T extends HTMLElement = HTMLElement>(id: string): T | null {
  if (typeof document === 'undefined') {
    return null;
  }
  return document.getElementById(id) as T | null;
}

// Query selector
export function querySelector<T extends HTMLElement = HTMLElement>(selector: string): T | null {
  if (typeof document === 'undefined') {
    return null;
  }
  return document.querySelector(selector) as T | null;
}

// Query selector all
export function querySelectorAll<T extends HTMLElement = HTMLElement>(selector: string): T[] {
  if (typeof document === 'undefined') {
    return [];
  }
  return Array.from(document.querySelectorAll(selector)) as T[];
}

// Scroll to element
export function scrollToElement(element: HTMLElement | string, behavior: ScrollBehavior = 'smooth') {
  const el = typeof element === 'string' ? getElementById(element) : element;
  if (el) {
    el.scrollIntoView({ behavior, block: 'start' });
  }
}

// Scroll to top
export function scrollToTop(behavior: ScrollBehavior = 'smooth') {
  if (typeof window === 'undefined') {
    return;
  }
  window.scrollTo({ top: 0, behavior });
}

// Get scroll position
export function getScrollPosition(): { x: number; y: number } {
  if (typeof window === 'undefined') {
    return { x: 0, y: 0 };
  }
  return {
    x: window.pageXOffset || document.documentElement.scrollLeft,
    y: window.pageYOffset || document.documentElement.scrollTop,
  };
}

// Check if element is visible
export function isElementVisible(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

// Get element position
export function getElementPosition(element: HTMLElement): { x: number; y: number; width: number; height: number } {
  const rect = element.getBoundingClientRect();
  return {
    x: rect.left + window.scrollX,
    y: rect.top + window.scrollY,
    width: rect.width,
    height: rect.height,
  };
}

// Add class to element
export function addClass(element: HTMLElement, className: string) {
  element.classList.add(className);
}

// Remove class from element
export function removeClass(element: HTMLElement, className: string) {
  element.classList.remove(className);
}

// Toggle class on element
export function toggleClass(element: HTMLElement, className: string) {
  element.classList.toggle(className);
}

// Check if element has class
export function hasClass(element: HTMLElement, className: string): boolean {
  return element.classList.contains(className);
}



