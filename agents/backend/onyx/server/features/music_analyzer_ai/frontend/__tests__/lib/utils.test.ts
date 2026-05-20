import { cn, formatDuration, formatBPM, formatPercentage, debounce } from '@/lib/utils';

describe('Utils', () => {
  describe('cn', () => {
    it('should merge class names', () => {
      expect(cn('class1', 'class2')).toBe('class1 class2');
    });

    it('should handle conditional classes', () => {
      expect(cn('class1', false && 'class2', 'class3')).toBe('class1 class3');
    });

    it('should handle undefined and null', () => {
      expect(cn('class1', undefined, null, 'class2')).toBe('class1 class2');
    });

    it('should handle empty strings', () => {
      expect(cn('class1', '', 'class2')).toBe('class1 class2');
    });

    it('should handle arrays', () => {
      expect(cn(['class1', 'class2'], 'class3')).toBe('class1 class2 class3');
    });

    it('should handle objects', () => {
      expect(cn({ class1: true, class2: false, class3: true })).toBe('class1 class3');
    });

    it('should merge Tailwind classes correctly', () => {
      // tailwind-merge should handle conflicting classes
      const result = cn('p-4', 'p-2');
      expect(result).toBeTruthy();
    });
  });

  describe('formatDuration', () => {
    it('should format duration correctly', () => {
      expect(formatDuration(200000)).toBe('3:20');
      expect(formatDuration(180000)).toBe('3:00');
      expect(formatDuration(60000)).toBe('1:00');
      expect(formatDuration(30000)).toBe('0:30');
    });

    it('should handle zero', () => {
      expect(formatDuration(0)).toBe('0:00');
    });

    it('should handle negative values', () => {
      expect(formatDuration(-1000)).toBe('0:00');
    });

    it('should handle very large durations', () => {
      expect(formatDuration(3600000)).toBe('60:00'); // 1 hour
      expect(formatDuration(3661000)).toBe('61:01'); // 1 hour 1 minute 1 second
    });

    it('should handle single digit seconds', () => {
      expect(formatDuration(5000)).toBe('0:05');
      expect(formatDuration(9000)).toBe('0:09');
    });

    it('should handle milliseconds less than a second', () => {
      expect(formatDuration(500)).toBe('0:00');
      expect(formatDuration(999)).toBe('0:00');
    });

    it('should handle very small values', () => {
      expect(formatDuration(1)).toBe('0:00');
    });
  });

  describe('formatBPM', () => {
    it('should format BPM correctly', () => {
      expect(formatBPM(120)).toBe('120 BPM');
      expect(formatBPM(140.5)).toBe('141 BPM');
      expect(formatBPM(90.2)).toBe('90 BPM');
    });

    it('should handle zero', () => {
      expect(formatBPM(0)).toBe('0 BPM');
    });

    it('should handle negative values', () => {
      expect(formatBPM(-50)).toBe('0 BPM');
    });

    it('should round decimal values', () => {
      expect(formatBPM(120.4)).toBe('120 BPM');
      expect(formatBPM(120.6)).toBe('121 BPM');
    });

    it('should handle very high BPM', () => {
      expect(formatBPM(200)).toBe('200 BPM');
      expect(formatBPM(300)).toBe('300 BPM');
    });

    it('should handle very low BPM', () => {
      expect(formatBPM(30)).toBe('30 BPM');
      expect(formatBPM(1)).toBe('1 BPM');
    });
  });

  describe('formatPercentage', () => {
    it('should format percentage correctly', () => {
      expect(formatPercentage(0.75)).toBe('75%');
      expect(formatPercentage(0.5)).toBe('50%');
      expect(formatPercentage(1)).toBe('100%');
      expect(formatPercentage(0)).toBe('0%');
    });

    it('should clamp values above 1', () => {
      expect(formatPercentage(1.5)).toBe('100%');
      expect(formatPercentage(2)).toBe('100%');
    });

    it('should clamp values below 0', () => {
      expect(formatPercentage(-0.5)).toBe('0%');
      expect(formatPercentage(-1)).toBe('0%');
    });

    it('should round decimal values', () => {
      expect(formatPercentage(0.751)).toBe('75%');
      expect(formatPercentage(0.756)).toBe('76%');
    });

    it('should handle edge cases', () => {
      expect(formatPercentage(0.001)).toBe('0%');
      expect(formatPercentage(0.999)).toBe('100%');
    });

    it('should handle NaN', () => {
      expect(formatPercentage(NaN)).toBe('NaN%');
    });

    it('should handle Infinity', () => {
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
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn();
      debouncedFn();
      debouncedFn();

      expect(mockFn).not.toHaveBeenCalled();

      jest.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should pass arguments to debounced function', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn('arg1', 'arg2');
      jest.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledWith('arg1', 'arg2');
    });

    it('should reset timer on each call', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn();
      jest.advanceTimersByTime(50);
      debouncedFn();
      jest.advanceTimersByTime(50);
      debouncedFn();
      jest.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should handle multiple debounced functions independently', () => {
      const mockFn1 = jest.fn();
      const mockFn2 = jest.fn();
      const debouncedFn1 = debounce(mockFn1, 100);
      const debouncedFn2 = debounce(mockFn2, 200);

      debouncedFn1();
      debouncedFn2();

      jest.advanceTimersByTime(100);
      expect(mockFn1).toHaveBeenCalledTimes(1);
      expect(mockFn2).not.toHaveBeenCalled();

      jest.advanceTimersByTime(100);
      expect(mockFn2).toHaveBeenCalledTimes(1);
    });

    it('should handle zero wait time', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 0);

      debouncedFn();
      jest.advanceTimersByTime(0);

      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should handle very long wait time', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 10000);

      debouncedFn();
      jest.advanceTimersByTime(9999);
      expect(mockFn).not.toHaveBeenCalled();

      jest.advanceTimersByTime(1);
      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should handle function with no arguments', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn();
      jest.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledWith();
    });

    it('should handle function with many arguments', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn(1, 2, 3, 4, 5);
      jest.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledWith(1, 2, 3, 4, 5);
    });

    it('should clear previous timeout on new call', () => {
      const mockFn = jest.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn('first');
      jest.advanceTimersByTime(50);
      debouncedFn('second');
      jest.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(mockFn).toHaveBeenCalledWith('second');
    });
  });
});

