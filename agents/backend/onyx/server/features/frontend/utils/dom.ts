export function scrollToElement(
  element: HTMLElement | string,
  options: ScrollIntoViewOptions = {}
) {
  const target =
    typeof element === 'string'
      ? document.querySelector(element)
      : element;

  if (target instanceof HTMLElement) {
    target.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
      ...options,
    });
  }
}

export function scrollToTop(options: ScrollToOptions = {}) {
  window.scrollTo({
    top: 0,
    behavior: 'smooth',
    ...options,
  });
}

export function scrollToBottom(options: ScrollToOptions = {}) {
  window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: 'smooth',
    ...options,
  });
}

export function getElementOffset(element: HTMLElement) {
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top + window.scrollY,
    left: rect.left + window.scrollX,
    width: rect.width,
    height: rect.height,
  };
}

export function isElementInViewport(element: HTMLElement) {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

export function getScrollPosition() {
  return {
    x: window.scrollX || document.documentElement.scrollLeft,
    y: window.scrollY || document.documentElement.scrollTop,
  };
}

export function setScrollPosition(x: number, y: number) {
  window.scrollTo(x, y);
}

export function getViewportSize() {
  return {
    width: window.innerWidth || document.documentElement.clientWidth,
    height: window.innerHeight || document.documentElement.clientHeight,
  };
}

