import { renderHook, act } from '@testing-library/react';
import { useDebounce } from '@/lib/hooks/use-debounce';

describe('useDebounce', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should return initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('initial', 500));
    expect(result.current).toBe('initial');
  });

  it('should debounce value updates', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: 'initial', delay: 500 },
      }
    );

    expect(result.current).toBe('initial');

    // Update value
    rerender({ value: 'updated', delay: 500 });
    expect(result.current).toBe('initial'); // Should still be initial

    // Fast-forward time
    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toBe('updated');
  });

  it('should reset timer on rapid updates', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: 'initial', delay: 500 },
      }
    );

    rerender({ value: 'update1', delay: 500 });
    act(() => {
      jest.advanceTimersByTime(300);
    });

    rerender({ value: 'update2', delay: 500 });
    act(() => {
      jest.advanceTimersByTime(300);
    });

    rerender({ value: 'update3', delay: 500 });
    expect(result.current).toBe('initial');

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toBe('update3');
  });

  it('should work with different delay values', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: 'initial', delay: 1000 },
      }
    );

    rerender({ value: 'updated', delay: 1000 });
    act(() => {
      jest.advanceTimersByTime(500);
    });
    expect(result.current).toBe('initial');

    act(() => {
      jest.advanceTimersByTime(500);
    });
    expect(result.current).toBe('updated');
  });

  it('should work with numbers', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: 0, delay: 500 },
      }
    );

    rerender({ value: 100, delay: 500 });
    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toBe(100);
  });

  it('should work with objects', () => {
    const initial = { name: 'test' };
    const updated = { name: 'updated' };

    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      {
        initialProps: { value: initial, delay: 500 },
      }
    );

    rerender({ value: updated, delay: 500 });
    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toEqual(updated);
  });
});

