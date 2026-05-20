/**
 * Intersection Observer Testing
 * 
 * Tests that verify Intersection Observer functionality including
 * visibility detection, lazy loading, and scroll animations.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock IntersectionObserver
class MockIntersectionObserver {
  callback: IntersectionObserverCallback;
  options?: IntersectionObserverInit;
  observedElements: Element[] = [];

  constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
    this.callback = callback;
    this.options = options;
  }

  observe(element: Element) {
    this.observedElements.push(element);
  }

  unobserve(element: Element) {
    const index = this.observedElements.indexOf(element);
    if (index > -1) {
      this.observedElements.splice(index, 1);
    }
  }

  disconnect() {
    this.observedElements = [];
  }

  simulateIntersection(element: Element, isIntersecting: boolean) {
    const entry: IntersectionObserverEntry = {
      target: element,
      isIntersecting,
      intersectionRatio: isIntersecting ? 1 : 0,
      boundingClientRect: element.getBoundingClientRect(),
      rootBounds: null,
      intersectionRect: isIntersecting ? element.getBoundingClientRect() : new DOMRect(),
      time: Date.now(),
    };
    
    this.callback([entry], this);
  }
}

(global as any).IntersectionObserver = MockIntersectionObserver;

describe('Intersection Observer Testing', () => {
  describe('Observer Creation', () => {
    it('should create intersection observer', () => {
      const callback = vi.fn();
      const observer = new IntersectionObserver(callback);
      
      expect(observer).toBeDefined();
      expect(observer.callback).toBe(callback);
    });

    it('should create observer with options', () => {
      const callback = vi.fn();
      const options = {
        root: document.body,
        rootMargin: '10px',
        threshold: 0.5,
      };
      
      const observer = new IntersectionObserver(callback, options);
      expect(observer.options).toEqual(options);
    });
  });

  describe('Observing Elements', () => {
    it('should observe element', () => {
      const observer = new IntersectionObserver(vi.fn());
      const element = document.createElement('div');
      
      observer.observe(element);
      expect(observer.observedElements).toContain(element);
    });

    it('should observe multiple elements', () => {
      const observer = new IntersectionObserver(vi.fn());
      const elements = [
        document.createElement('div'),
        document.createElement('div'),
        document.createElement('div'),
      ];
      
      elements.forEach(el => observer.observe(el));
      expect(observer.observedElements).toHaveLength(3);
    });

    it('should unobserve element', () => {
      const observer = new IntersectionObserver(vi.fn());
      const element = document.createElement('div');
      
      observer.observe(element);
      observer.unobserve(element);
      expect(observer.observedElements).not.toContain(element);
    });

    it('should disconnect observer', () => {
      const observer = new IntersectionObserver(vi.fn());
      const elements = [
        document.createElement('div'),
        document.createElement('div'),
      ];
      
      elements.forEach(el => observer.observe(el));
      observer.disconnect();
      expect(observer.observedElements).toHaveLength(0);
    });
  });

  describe('Intersection Detection', () => {
    it('should detect when element enters viewport', (done) => {
      const callback = vi.fn((entries) => {
        expect(entries[0].isIntersecting).toBe(true);
        done();
      });
      
      const observer = new IntersectionObserver(callback);
      const element = document.createElement('div');
      observer.observe(element);
      observer.simulateIntersection(element, true);
    });

    it('should detect when element leaves viewport', (done) => {
      const callback = vi.fn((entries) => {
        expect(entries[0].isIntersecting).toBe(false);
        done();
      });
      
      const observer = new IntersectionObserver(callback);
      const element = document.createElement('div');
      observer.observe(element);
      observer.simulateIntersection(element, false);
    });
  });

  describe('Lazy Loading', () => {
    it('should lazy load images', () => {
      const lazyLoadImage = (img: HTMLImageElement, observer: IntersectionObserver) => {
        observer.observe(img);
        
        const handleIntersection = (entries: IntersectionObserverEntry[]) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              const image = entry.target as HTMLImageElement;
              if (image.dataset.src) {
                image.src = image.dataset.src;
                observer.unobserve(image);
              }
            }
          });
        };
        
        return handleIntersection;
      };
      
      const img = document.createElement('img');
      img.dataset.src = '/image.jpg';
      const observer = new IntersectionObserver(vi.fn());
      const handler = lazyLoadImage(img, observer);
      
      expect(observer.observedElements).toContain(img);
    });
  });

  describe('Scroll Animations', () => {
    it('should trigger animation on scroll', () => {
      const animateOnScroll = (element: HTMLElement, observer: IntersectionObserver) => {
        observer.observe(element);
        
        const handleIntersection = (entries: IntersectionObserverEntry[]) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              entry.target.classList.add('animate');
            } else {
              entry.target.classList.remove('animate');
            }
          });
        };
        
        return handleIntersection;
      };
      
      const element = document.createElement('div');
      const observer = new IntersectionObserver(vi.fn());
      const handler = animateOnScroll(element, observer);
      
      observer.simulateIntersection(element, true);
      expect(element.classList.contains('animate')).toBe(true);
    });
  });

  describe('Threshold Options', () => {
    it('should use threshold for intersection detection', () => {
      const callback = vi.fn();
      const options = { threshold: [0, 0.5, 1] };
      const observer = new IntersectionObserver(callback, options);
      
      expect(observer.options?.threshold).toEqual([0, 0.5, 1]);
    });
  });
});

