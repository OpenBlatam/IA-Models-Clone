// Utility functions tests
import { formatDuration, formatBPM, formatPercentage, debounce, cn } from '@/lib/utils';

describe('Utility Functions', () => {
  describe('cn', () => {
    it('should merge class names correctly', () => {
      expect(cn('foo', 'bar')).toBe('foo bar');
      expect(cn('foo', false && 'bar', 'baz')).toBe('foo baz');
      expect(cn('px-2 py-1', 'px-4')).toBe('px-4 py-1'); // tailwind-merge should override
    });

    it('should handle conditional classes', () => {
      const isActive = true;
      expect(cn('base', isActive && 'active', 'other')).toBe('base active other');
    });

    it('should handle null and undefined values', () => {
      expect(cn('foo', null, 'bar', undefined, 'baz')).toBe('foo bar baz');
    });

    it('should handle empty strings', () => {
      expect(cn('foo', '', 'bar')).toBe('foo bar');
    });

    it('should handle arrays of classes', () => {
      expect(cn(['foo', 'bar'], 'baz')).toBe('foo bar baz');
    });

    it('should handle objects with boolean values', () => {
      expect(cn({ foo: true, bar: false, baz: true })).toContain('foo');
      expect(cn({ foo: true, bar: false, baz: true })).toContain('baz');
      expect(cn({ foo: true, bar: false, baz: true })).not.toContain('bar');
    });
  });

  describe('formatDuration', () => {
    it('should format milliseconds to mm:ss', () => {
      expect(formatDuration(0)).toBe('0:00');
      expect(formatDuration(1000)).toBe('0:01');
      expect(formatDuration(60000)).toBe('1:00');
      expect(formatDuration(125000)).toBe('2:05');
      expect(formatDuration(3661000)).toBe('61:01');
    });

    it('should handle negative values', () => {
      expect(formatDuration(-1000)).toBe('0:00');
      expect(formatDuration(-5000)).toBe('0:00');
    });

    it('should handle large durations', () => {
      expect(formatDuration(3600000)).toBe('60:00'); // 1 hour
      expect(formatDuration(7200000)).toBe('120:00'); // 2 hours
    });

    it('should handle very small durations', () => {
      expect(formatDuration(1)).toBe('0:00');
      expect(formatDuration(999)).toBe('0:00');
    });

    it('should handle NaN and Infinity', () => {
      expect(formatDuration(NaN)).toBe('0:00');
      expect(formatDuration(Infinity)).toBe('0:00');
    });

    it('should handle decimal milliseconds', () => {
      expect(formatDuration(1250.5)).toBe('0:01');
      expect(formatDuration(59999.9)).toBe('0:59');
    });
  });

  describe('formatBPM', () => {
    it('should format BPM values', () => {
      expect(formatBPM(120)).toBe('120 BPM');
      expect(formatBPM(120.5)).toBe('121 BPM');
      expect(formatBPM(0)).toBe('0 BPM');
    });

    it('should handle negative values', () => {
      expect(formatBPM(-10)).toBe('0 BPM');
    });

    it('should round decimal values', () => {
      expect(formatBPM(120.4)).toBe('120 BPM');
      expect(formatBPM(120.6)).toBe('121 BPM');
    });

    it('should handle very large BPM values', () => {
      expect(formatBPM(200)).toBe('200 BPM');
      expect(formatBPM(300.7)).toBe('301 BPM');
    });

    it('should handle NaN and Infinity', () => {
      expect(formatBPM(NaN)).toBe('0 BPM');
      expect(formatBPM(Infinity)).toBe('0 BPM');
    });
  });

  describe('formatPercentage', () => {
    it('should format decimal values as percentages', () => {
      expect(formatPercentage(0)).toBe('0%');
      expect(formatPercentage(0.5)).toBe('50%');
      expect(formatPercentage(1)).toBe('100%');
      expect(formatPercentage(0.75)).toBe('75%');
    });

    it('should clamp values between 0 and 1', () => {
      expect(formatPercentage(-0.5)).toBe('0%');
      expect(formatPercentage(1.5)).toBe('100%');
      expect(formatPercentage(2)).toBe('100%');
    });

    it('should round percentage values', () => {
      expect(formatPercentage(0.333)).toBe('33%');
      expect(formatPercentage(0.666)).toBe('67%');
    });

    it('should handle edge cases near boundaries', () => {
      expect(formatPercentage(0.001)).toBe('0%');
      expect(formatPercentage(0.999)).toBe('100%');
      expect(formatPercentage(0.004)).toBe('0%');
      expect(formatPercentage(0.995)).toBe('100%');
    });

    it('should handle NaN and Infinity', () => {
      expect(formatPercentage(NaN)).toBe('0%');
      expect(formatPercentage(Infinity)).toBe('100%');
      expect(formatPercentage(-Infinity)).toBe('0%');
    });
  });

  describe('debounce', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should debounce function calls', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 300);

      debouncedFn('call1');
      debouncedFn('call2');
      debouncedFn('call3');

      expect(mockFn).not.toHaveBeenCalled();

      jest.advanceTimersByTime(300);

      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(mockFn).toHaveBeenCalledWith('call3');
    });

    it('should reset timer on each call', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 300);

      debouncedFn('call1');
      jest.advanceTimersByTime(200);
      debouncedFn('call2');
      jest.advanceTimersByTime(200);
      debouncedFn('call3');

      expect(mockFn).not.toHaveBeenCalled();

      jest.advanceTimersByTime(300);

      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(mockFn).toHaveBeenCalledWith('call3');
    });

    it('should handle multiple arguments', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 300);

      debouncedFn('arg1', 'arg2', 'arg3');
      jest.advanceTimersByTime(300);

      expect(mockFn).toHaveBeenCalledWith('arg1', 'arg2', 'arg3');
    });
  });
});

