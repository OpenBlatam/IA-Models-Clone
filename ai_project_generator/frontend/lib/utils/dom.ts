export const domUtils = {
  scrollTo: (element: HTMLElement | string, options?: ScrollIntoViewOptions) => {
    const target = typeof element === 'string' ? document.querySelector(element) : element
    if (target instanceof HTMLElement) {
      target.scrollIntoView({ behavior: 'smooth', ...options })
    }
  },

  scrollToTop: (options?: ScrollIntoViewOptions) => {
    window.scrollTo({ top: 0, behavior: 'smooth', ...options })
  },

  getScrollPosition: (): { x: number; y: number } => {
    return {
      x: window.pageXOffset || document.documentElement.scrollLeft,
      y: window.pageYOffset || document.documentElement.scrollTop,
    }
  },

  isElementInViewport: (element: HTMLElement): boolean => {
    const rect = element.getBoundingClientRect()
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    )
  },

  getElementOffset: (element: HTMLElement): { top: number; left: number } => {
    const rect = element.getBoundingClientRect()
    return {
      top: rect.top + window.pageYOffset,
      left: rect.left + window.pageXOffset,
    }
  },

  focusElement: (element: HTMLElement | string) => {
    const target = typeof element === 'string' ? document.querySelector(element) : element
    if (target instanceof HTMLElement) {
      target.focus()
    }
  },
}

