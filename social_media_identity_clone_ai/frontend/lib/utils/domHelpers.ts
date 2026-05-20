export const scrollToElement = (element: HTMLElement | string, options?: ScrollIntoViewOptions): void => {
  const target = typeof element === 'string' ? document.querySelector(element) : element;
  if (target instanceof HTMLElement) {
    target.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
      ...options,
    });
  }
};

export const scrollToTop = (behavior: ScrollBehavior = 'smooth'): void => {
  window.scrollTo({ top: 0, behavior });
};

export const scrollToBottom = (behavior: ScrollBehavior = 'smooth'): void => {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior });
};

export const getElementOffset = (element: HTMLElement): { top: number; left: number } => {
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top + window.scrollY,
    left: rect.left + window.scrollX,
  };
};

export const isElementInViewport = (element: HTMLElement, partial = false): boolean => {
  const rect = element.getBoundingClientRect();
  const windowHeight = window.innerHeight || document.documentElement.clientHeight;
  const windowWidth = window.innerWidth || document.documentElement.clientWidth;

  if (partial) {
    return (
      rect.top < windowHeight &&
      rect.bottom > 0 &&
      rect.left < windowWidth &&
      rect.right > 0
    );
  }

  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= windowHeight &&
    rect.right <= windowWidth
  );
};

export const getScrollPosition = (): { x: number; y: number } => {
  return {
    x: window.pageXOffset || document.documentElement.scrollLeft,
    y: window.pageYOffset || document.documentElement.scrollTop,
  };
};

export const getViewportSize = (): { width: number; height: number } => {
  return {
    width: window.innerWidth || document.documentElement.clientWidth,
    height: window.innerHeight || document.documentElement.clientHeight,
  };
};

export const addClass = (element: HTMLElement, className: string): void => {
  element.classList.add(className);
};

export const removeClass = (element: HTMLElement, className: string): void => {
  element.classList.remove(className);
};

export const toggleClass = (element: HTMLElement, className: string): void => {
  element.classList.toggle(className);
};

export const hasClass = (element: HTMLElement, className: string): boolean => {
  return element.classList.contains(className);
};



