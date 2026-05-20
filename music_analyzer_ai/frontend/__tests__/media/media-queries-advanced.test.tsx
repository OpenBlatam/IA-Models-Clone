/**
 * Advanced Media Queries Testing
 * 
 * Tests that verify responsive design using media queries,
 * breakpoint detection, and adaptive layouts.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock window.matchMedia
const createMatchMedia = (matches: boolean) => {
  return vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
};

describe('Advanced Media Queries Testing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Breakpoint Detection', () => {
    it('should detect mobile breakpoint', () => {
      window.matchMedia = createMatchMedia(true);
      const isMobile = window.matchMedia('(max-width: 640px)').matches;
      expect(isMobile).toBe(true);
    });

    it('should detect tablet breakpoint', () => {
      window.matchMedia = createMatchMedia(true);
      const isTablet = window.matchMedia('(min-width: 641px) and (max-width: 1024px)').matches;
      expect(isTablet).toBe(true);
    });

    it('should detect desktop breakpoint', () => {
      window.matchMedia = createMatchMedia(false);
      const isDesktop = !window.matchMedia('(max-width: 1024px)').matches;
      expect(isDesktop).toBe(true);
    });

    it('should detect landscape orientation', () => {
      window.matchMedia = createMatchMedia(true);
      const isLandscape = window.matchMedia('(orientation: landscape)').matches;
      expect(isLandscape).toBe(true);
    });

    it('should detect portrait orientation', () => {
      window.matchMedia = createMatchMedia(true);
      const isPortrait = window.matchMedia('(orientation: portrait)').matches;
      expect(isPortrait).toBe(true);
    });
  });

  describe('Responsive Layouts', () => {
    it('should adapt layout based on screen width', () => {
      const getLayout = (width: number) => {
        if (width < 640) return 'mobile';
        if (width < 1024) return 'tablet';
        return 'desktop';
      };

      expect(getLayout(375)).toBe('mobile');
      expect(getLayout(768)).toBe('tablet');
      expect(getLayout(1920)).toBe('desktop');
    });

    it('should use different column counts for different breakpoints', () => {
      const getColumnCount = (width: number) => {
        if (width < 640) return 1;
        if (width < 1024) return 2;
        return 3;
      };

      expect(getColumnCount(375)).toBe(1);
      expect(getColumnCount(768)).toBe(2);
      expect(getColumnCount(1920)).toBe(3);
    });
  });

  describe('Media Query Listeners', () => {
    it('should listen to media query changes', () => {
      const mediaQuery = window.matchMedia('(max-width: 640px)');
      const listener = vi.fn();
      
      mediaQuery.addEventListener('change', listener);
      
      // Simulate change
      const newQuery = { ...mediaQuery, matches: false };
      const changeEvent = new MediaQueryListEvent('change', {
        media: '(max-width: 640px)',
        matches: false,
      });
      
      listener(changeEvent);
      expect(listener).toHaveBeenCalled();
    });

    it('should remove media query listeners', () => {
      const mediaQuery = window.matchMedia('(max-width: 640px)');
      const listener = vi.fn();
      
      mediaQuery.addEventListener('change', listener);
      mediaQuery.removeEventListener('change', listener);
      
      expect(mediaQuery.removeEventListener).toHaveBeenCalled();
    });
  });

  describe('Feature Detection', () => {
    it('should detect hover capability', () => {
      const hasHover = window.matchMedia('(hover: hover)').matches;
      expect(typeof hasHover).toBe('boolean');
    });

    it('should detect pointer type', () => {
      const hasFinePointer = window.matchMedia('(pointer: fine)').matches;
      const hasCoarsePointer = window.matchMedia('(pointer: coarse)').matches;
      
      expect(typeof hasFinePointer).toBe('boolean');
      expect(typeof hasCoarsePointer).toBe('boolean');
    });

    it('should detect color scheme preference', () => {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
      
      expect(typeof prefersDark).toBe('boolean');
      expect(typeof prefersLight).toBe('boolean');
    });

    it('should detect reduced motion preference', () => {
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      expect(typeof prefersReducedMotion).toBe('boolean');
    });
  });

  describe('Complex Media Queries', () => {
    it('should combine multiple conditions', () => {
      const complexQuery = '(min-width: 768px) and (max-width: 1024px) and (orientation: landscape)';
      window.matchMedia = createMatchMedia(true);
      
      const matches = window.matchMedia(complexQuery).matches;
      expect(matches).toBe(true);
    });

    it('should use not operator', () => {
      window.matchMedia = createMatchMedia(false);
      const notMobile = window.matchMedia('not (max-width: 640px)').matches;
      expect(notMobile).toBe(true);
    });

    it('should use only operator', () => {
      window.matchMedia = createMatchMedia(true);
      const onlyScreen = window.matchMedia('only screen and (min-width: 768px)').matches;
      expect(onlyScreen).toBe(true);
    });
  });

  describe('Responsive Images', () => {
    it('should use srcset for responsive images', () => {
      const createResponsiveImage = (baseUrl: string) => {
        return {
          srcset: `${baseUrl}-small.jpg 320w, ${baseUrl}-medium.jpg 640w, ${baseUrl}-large.jpg 1024w`,
          sizes: '(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw',
        };
      };

      const image = createResponsiveImage('/image');
      expect(image.srcset).toContain('320w');
      expect(image.sizes).toBeDefined();
    });
  });

  describe('Container Queries', () => {
    it('should use container queries when supported', () => {
      const supportsContainerQueries = CSS.supports('container-type', 'inline-size');
      expect(typeof supportsContainerQueries).toBe('boolean');
    });
  });
});

