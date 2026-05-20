/**
 * Performance utilities for monitoring and optimization
 */

// Performance monitoring
export function measurePerformance(name: string, fn: () => void) {
  if (typeof window !== 'undefined' && window.performance) {
    const start = performance.now();
    fn();
    const end = performance.now();
    const duration = end - start;
    
    if (duration > 100) {
      console.warn(`[Performance] ${name} took ${duration.toFixed(2)}ms`);
    }
    
    return duration;
  }
  fn();
  return 0;
}

// Debounce function
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

// Throttle function
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;

  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

// Lazy load images
export function lazyLoadImage(img: HTMLImageElement, src: string) {
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          img.src = src;
          observer.unobserve(img);
        }
      });
    });

    imageObserver.observe(img);
  } else {
    // Fallback for browsers without IntersectionObserver
    img.src = src;
  }
}

// Check if element is in viewport
export function isInViewport(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

// Preload resources
export function preloadResource(href: string, as: string) {
  if (typeof document !== 'undefined') {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
  }
}

// Get performance metrics
export function getPerformanceMetrics() {
  if (typeof window === 'undefined' || !window.performance) {
    return null;
  }

  const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
  
  if (!navigation) {
    return null;
  }

  return {
    dns: navigation.domainLookupEnd - navigation.domainLookupStart,
    tcp: navigation.connectEnd - navigation.connectStart,
    request: navigation.responseStart - navigation.requestStart,
    response: navigation.responseEnd - navigation.responseStart,
    dom: navigation.domContentLoadedEventEnd - navigation.responseEnd,
    load: navigation.loadEventEnd - navigation.loadEventStart,
    total: navigation.loadEventEnd - navigation.fetchStart,
  };
}
