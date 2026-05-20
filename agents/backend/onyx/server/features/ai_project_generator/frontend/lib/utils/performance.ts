export const performanceUtils = {
  measure: (name: string, fn: () => void) => {
    if (typeof window !== 'undefined' && window.performance) {
      performance.mark(`${name}-start`)
      fn()
      performance.mark(`${name}-end`)
      performance.measure(name, `${name}-start`, `${name}-end`)
      const measure = performance.getEntriesByName(name)[0]
      console.log(`${name}: ${measure.duration.toFixed(2)}ms`)
    } else {
      fn()
    }
  },

  debounce: <T extends (...args: unknown[]) => unknown>(fn: T, delay: number) => {
    let timeoutId: NodeJS.Timeout
    return (...args: Parameters<T>) => {
      clearTimeout(timeoutId)
      timeoutId = setTimeout(() => fn(...args), delay)
    }
  },

  throttle: <T extends (...args: unknown[]) => unknown>(fn: T, limit: number) => {
    let inThrottle: boolean
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        fn(...args)
        inThrottle = true
        setTimeout(() => (inThrottle = false), limit)
      }
    }
  },

  requestIdleCallback: (callback: () => void, options?: { timeout?: number }) => {
    if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
      return window.requestIdleCallback(callback, options)
    } else {
      return setTimeout(callback, 1)
    }
  },
}

