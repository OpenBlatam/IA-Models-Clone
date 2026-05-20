/**
 * Performance Optimizer Module
 * ============================
 * Optimizes performance through lazy loading, debouncing, throttling, and resource management
 */

const PerformanceOptimizer = {
    /**
     * Lazy load images
     */
    lazyLoadImages() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.removeAttribute('data-src');
                            observer.unobserve(img);
                        }
                    }
                });
            });

            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        } else {
            // Fallback for browsers without IntersectionObserver
            document.querySelectorAll('img[data-src]').forEach(img => {
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
            });
        }
    },

    /**
     * Optimize image loading with preloading
     */
    preloadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    },

    /**
     * Batch DOM updates using requestAnimationFrame
     */
    batchDOMUpdates(updates) {
        return new Promise(resolve => {
            requestAnimationFrame(() => {
                updates.forEach(update => update());
                resolve();
            });
        });
    },

    /**
     * Debounce with immediate option (delegates to Debounce module)
     */
    debounce(func, wait, immediate = false) {
        if (typeof Debounce !== 'undefined') {
            return Debounce.debounce(func, wait, immediate);
        }
        // Fallback implementation
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    },

    /**
     * Throttle with leading and trailing options
     */
    throttle(func, limit, options = {}) {
        let inThrottle;
        let lastFunc;
        let lastRan;
        const { leading = true, trailing = true } = options;

        return function executedFunction(...args) {
            if (!inThrottle) {
                if (leading) {
                    func.apply(this, args);
                }
                lastRan = Date.now();
                inThrottle = true;
            } else {
                clearTimeout(lastFunc);
                lastFunc = setTimeout(() => {
                    if (Date.now() - lastRan >= limit) {
                        if (trailing) {
                            func.apply(this, args);
                        }
                        lastRan = Date.now();
                    }
                }, Math.max(limit - (Date.now() - lastRan), 0));
            }
        };
    },

    /**
     * Memoize function results
     */
    memoize(func, keyGenerator = null) {
        const cache = new Map();
        return function memoizedFunction(...args) {
            const key = keyGenerator ? keyGenerator(...args) : JSON.stringify(args);
            if (cache.has(key)) {
                return cache.get(key);
            }
            const result = func.apply(this, args);
            cache.set(key, result);
            return result;
        };
    },

    /**
     * Virtual scrolling for large lists
     */
    createVirtualScroller(container, items, itemHeight, renderItem) {
        const visibleCount = Math.ceil(container.clientHeight / itemHeight);
        let startIndex = 0;
        let endIndex = visibleCount;

        const updateVisibleItems = () => {
            const scrollTop = container.scrollTop;
            startIndex = Math.floor(scrollTop / itemHeight);
            endIndex = Math.min(startIndex + visibleCount + 1, items.length);

            const visibleItems = items.slice(startIndex, endIndex);
            const offsetY = startIndex * itemHeight;
            const totalHeight = items.length * itemHeight;

            container.innerHTML = '';
            const wrapper = document.createElement('div');
            wrapper.style.height = `${totalHeight}px`;
            wrapper.style.position = 'relative';

            visibleItems.forEach((item, index) => {
                const element = renderItem(item, startIndex + index);
                element.style.position = 'absolute';
                element.style.top = `${(startIndex + index) * itemHeight}px`;
                element.style.height = `${itemHeight}px`;
                wrapper.appendChild(element);
            });

            container.appendChild(wrapper);
        };

        container.addEventListener('scroll', this.throttle(updateVisibleItems, 16));
        updateVisibleItems();
    },

    /**
     * Optimize canvas operations
     */
    optimizeCanvas(canvas, width, height) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        
        return ctx;
    },

    /**
     * Compress image data
     */
    compressImage(file, maxWidth = 1920, maxHeight = 1080, quality = 0.8) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    let width = img.width;
                    let height = img.height;

                    if (width > height) {
                        if (width > maxWidth) {
                            height *= maxWidth / width;
                            width = maxWidth;
                        }
                    } else {
                        if (height > maxHeight) {
                            width *= maxHeight / height;
                            height = maxHeight;
                        }
                    }

                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);

                    canvas.toBlob(resolve, file.type, quality);
                };
                img.onerror = reject;
                img.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    /**
     * Measure performance
     */
    measurePerformance(name, fn) {
        if (typeof performance !== 'undefined' && performance.mark) {
            performance.mark(`${name}-start`);
            const result = fn();
            performance.mark(`${name}-end`);
            performance.measure(name, `${name}-start`, `${name}-end`);
            const measure = performance.getEntriesByName(name)[0];
            if (typeof Logger !== 'undefined') {
                Logger.debug(`Performance: ${name} took ${measure.duration.toFixed(2)}ms`);
            }
            return result;
        }
        return fn();
    },

    /**
     * Optimize event listeners with delegation
     */
    delegateEvents(container, selector, eventType, handler) {
        container.addEventListener(eventType, (e) => {
            const target = e.target.closest(selector);
            if (target) {
                handler.call(target, e);
            }
        });
    },

    /**
     * Clean up unused resources
     */
    cleanup() {
        // Clear caches
        if (typeof Cache !== 'undefined') {
            Cache.clearExpired();
        }

        // Clear unused event listeners
        if (typeof EventBus !== 'undefined') {
            // EventBus cleanup can be handled by the module itself
        }

        // Force garbage collection hint (if available)
        if (window.gc) {
            window.gc();
        }
    },

    /**
     * Initialize performance optimizations
     */
    init() {
        // Lazy load images
        this.lazyLoadImages();

        // Setup periodic cleanup
        setInterval(() => {
            this.cleanup();
        }, 60000); // Every minute

        // Monitor performance
        if (typeof PerformanceObserver !== 'undefined') {
            try {
                const observer = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (entry.entryType === 'measure' && typeof Logger !== 'undefined') {
                            Logger.debug(`Performance: ${entry.name} - ${entry.duration.toFixed(2)}ms`);
                        }
                    }
                });
                observer.observe({ entryTypes: ['measure'] });
            } catch (e) {
                // PerformanceObserver not supported
            }
        }

        if (typeof Logger !== 'undefined') {
            Logger.info('Performance optimizer initialized');
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => PerformanceOptimizer.init());
    } else {
        PerformanceOptimizer.init();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceOptimizer;
}
