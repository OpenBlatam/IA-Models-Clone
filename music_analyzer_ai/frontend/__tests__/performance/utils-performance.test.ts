/**
 * Performance Tests - Utils
 * Tests to ensure utility functions perform well
 */

import { formatDuration, formatBPM, formatPercentage, debounce } from '@/lib/utils';

describe('Utils Performance', () => {
  describe('formatDuration performance', () => {
    it('should format 1000 durations quickly', () => {
      const start = performance.now();
      for (let i = 0; i < 1000; i++) {
        formatDuration(i * 1000);
      }
      const end = performance.now();
      const duration = end - start;

      // Should complete in less than 100ms
      expect(duration).toBeLessThan(100);
    });

    it('should handle large batches efficiently', () => {
      const start = performance.now();
      for (let i = 0; i < 10000; i++) {
        formatDuration(Math.random() * 3600000);
      }
      const end = performance.now();
      const duration = end - start;

      // Should complete in less than 500ms
      expect(duration).toBeLessThan(500);
    });
  });

  describe('formatBPM performance', () => {
    it('should format 1000 BPM values quickly', () => {
      const start = performance.now();
      for (let i = 0; i < 1000; i++) {
        formatBPM(Math.random() * 200);
      }
      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(100);
    });
  });

  describe('formatPercentage performance', () => {
    it('should format 1000 percentages quickly', () => {
      const start = performance.now();
      for (let i = 0; i < 1000; i++) {
        formatPercentage(Math.random());
      }
      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(100);
    });
  });

  describe('debounce performance', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should handle many rapid calls efficiently', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 100);

      const start = performance.now();
      for (let i = 0; i < 1000; i++) {
        debouncedFn();
      }
      const end = performance.now();
      const duration = end - start;

      // Should handle 1000 calls quickly
      expect(duration).toBeLessThan(100);

      jest.advanceTimersByTime(100);
      expect(mockFn).toHaveBeenCalledTimes(1);
    });
  });
});

