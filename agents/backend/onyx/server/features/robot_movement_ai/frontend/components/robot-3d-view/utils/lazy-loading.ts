/**
 * Intelligent lazy loading utilities
 * @module robot-3d-view/utils/lazy-loading
 */

/**
 * Lazy loading options
 */
export interface LazyLoadOptions {
  root?: Element | null;
  rootMargin?: string;
  threshold?: number | number[];
  once?: boolean;
}

/**
 * Lazy loading entry
 */
interface LazyLoadEntry {
  element: Element;
  callback: () => void;
  observer?: IntersectionObserver;
}

/**
 * Lazy Loading Manager class
 */
export class LazyLoadingManager {
  private entries: Map<Element, LazyLoadEntry> = new Map();
  private defaultObserver?: IntersectionObserver;
  private defaultOptions: LazyLoadOptions = {
    rootMargin: '50px',
    threshold: 0.01,
    once: true,
  };

  /**
   * Lazy loads an element
   */
  lazyLoad(
    element: Element,
    callback: () => void,
    options: LazyLoadOptions = {}
  ): () => void {
    const mergedOptions = { ...this.defaultOptions, ...options };
    const entry: LazyLoadEntry = {
      element,
      callback,
    };

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            callback();
            if (mergedOptions.once) {
              observer.unobserve(entry.target);
              this.entries.delete(entry.target);
            }
          }
        });
      },
      {
        root: mergedOptions.root,
        rootMargin: mergedOptions.rootMargin,
        threshold: mergedOptions.threshold,
      }
    );

    observer.observe(element);
    entry.observer = observer;
    this.entries.set(element, entry);

    // Return cleanup function
    return () => {
      observer.unobserve(element);
      this.entries.delete(element);
    };
  }

  /**
   * Lazy loads an image
   */
  lazyLoadImage(
    img: HTMLImageElement,
    src: string,
    options: LazyLoadOptions = {}
  ): () => void {
    return this.lazyLoad(
      img,
      () => {
        img.src = src;
        img.loading = 'lazy';
      },
      options
    );
  }

  /**
   * Lazy loads a script
   */
  lazyLoadScript(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      // Check if script already loaded
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
      document.head.appendChild(script);
    });
  }

  /**
   * Lazy loads a stylesheet
   */
  lazyLoadStylesheet(href: string): Promise<void> {
    return new Promise((resolve, reject) => {
      // Check if stylesheet already loaded
      const existing = document.querySelector(`link[href="${href}"]`);
      if (existing) {
        resolve();
        return;
      }

      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      link.onload = () => resolve();
      link.onerror = () => reject(new Error(`Failed to load stylesheet: ${href}`));
      document.head.appendChild(link);
    });
  }

  /**
   * Preloads a resource
   */
  preloadResource(href: string, as: string): void {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
  }

  /**
   * Prefetches a resource
   */
  prefetchResource(href: string): void {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = href;
    document.head.appendChild(link);
  }

  /**
   * Cleans up all lazy loading entries
   */
  cleanup(): void {
    this.entries.forEach((entry) => {
      entry.observer?.disconnect();
    });
    this.entries.clear();
  }
}

/**
 * Global lazy loading manager instance
 */
export const lazyLoadingManager = new LazyLoadingManager();



