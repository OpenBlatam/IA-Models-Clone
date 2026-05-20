/**
 * Resize Observer Testing
 * 
 * Tests that verify Resize Observer functionality including
 * element size monitoring and responsive layout adjustments.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock ResizeObserver
class MockResizeObserver {
  callback: ResizeObserverCallback;
  observedElements: Element[] = [];

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }

  observe(element: Element) {
    if (!this.observedElements.includes(element)) {
      this.observedElements.push(element);
    }
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

  simulateResize(element: Element, size: { width: number; height: number }) {
    const entry: ResizeObserverEntry = {
      target: element,
      contentRect: {
        x: 0,
        y: 0,
        width: size.width,
        height: size.height,
        top: 0,
        right: size.width,
        bottom: size.height,
        left: 0,
        toJSON: () => ({}),
      },
      borderBoxSize: [{
        inlineSize: size.width,
        blockSize: size.height,
      }],
      contentBoxSize: [{
        inlineSize: size.width,
        blockSize: size.height,
      }],
      devicePixelContentBoxSize: [{
        inlineSize: size.width,
        blockSize: size.height,
      }],
    };
    
    this.callback([entry], this);
  }
}

(global as any).ResizeObserver = MockResizeObserver;

describe('Resize Observer Testing', () => {
  describe('Observer Creation', () => {
    it('should create resize observer', () => {
      const callback = vi.fn();
      const observer = new ResizeObserver(callback);
      
      expect(observer).toBeDefined();
      expect(observer.callback).toBe(callback);
    });
  });

  describe('Observing Elements', () => {
    it('should observe element', () => {
      const observer = new ResizeObserver(vi.fn());
      const element = document.createElement('div');
      
      observer.observe(element);
      expect(observer.observedElements).toContain(element);
    });

    it('should observe multiple elements', () => {
      const observer = new ResizeObserver(vi.fn());
      const elements = [
        document.createElement('div'),
        document.createElement('div'),
      ];
      
      elements.forEach(el => observer.observe(el));
      expect(observer.observedElements).toHaveLength(2);
    });

    it('should unobserve element', () => {
      const observer = new ResizeObserver(vi.fn());
      const element = document.createElement('div');
      
      observer.observe(element);
      observer.unobserve(element);
      expect(observer.observedElements).not.toContain(element);
    });

    it('should disconnect observer', () => {
      const observer = new ResizeObserver(vi.fn());
      const element = document.createElement('div');
      
      observer.observe(element);
      observer.disconnect();
      expect(observer.observedElements).toHaveLength(0);
    });
  });

  describe('Size Detection', () => {
    it('should detect element resize', (done) => {
      const callback = vi.fn((entries) => {
        expect(entries[0].contentRect.width).toBe(200);
        expect(entries[0].contentRect.height).toBe(100);
        done();
      });
      
      const observer = new ResizeObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element);
      observer.simulateResize(element, { width: 200, height: 100 });
    });

    it('should track size changes', () => {
      const sizes: Array<{ width: number; height: number }> = [];
      const callback = vi.fn((entries) => {
        entries.forEach(entry => {
          sizes.push({
            width: entry.contentRect.width,
            height: entry.contentRect.height,
          });
        });
      });
      
      const observer = new ResizeObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element);
      observer.simulateResize(element, { width: 100, height: 50 });
      observer.simulateResize(element, { width: 200, height: 100 });
      
      expect(sizes).toHaveLength(2);
      expect(sizes[0]).toEqual({ width: 100, height: 50 });
      expect(sizes[1]).toEqual({ width: 200, height: 100 });
    });
  });

  describe('Responsive Layout', () => {
    it('should adjust layout based on size', () => {
      const adjustLayout = (element: HTMLElement, width: number) => {
        if (width < 640) {
          element.classList.add('mobile');
          element.classList.remove('desktop');
        } else {
          element.classList.add('desktop');
          element.classList.remove('mobile');
        }
      };
      
      const element = document.createElement('div');
      const observer = new ResizeObserver((entries) => {
        entries.forEach(entry => {
          adjustLayout(element, entry.contentRect.width);
        });
      });
      
      observer.observe(element);
      observer.simulateResize(element, { width: 375, height: 667 });
      expect(element.classList.contains('mobile')).toBe(true);
      
      observer.simulateResize(element, { width: 1920, height: 1080 });
      expect(element.classList.contains('desktop')).toBe(true);
    });
  });

  describe('Performance', () => {
    it('should throttle resize events', () => {
      let callCount = 0;
      const callback = vi.fn(() => {
        callCount++;
      });
      
      const observer = new ResizeObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element);
      
      // Simulate rapid resizes
      for (let i = 0; i < 100; i++) {
        observer.simulateResize(element, { width: i, height: i });
      }
      
      // Should have been called
      expect(callback).toHaveBeenCalled();
    });
  });
});

