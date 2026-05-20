import { renderHook, act } from '@testing-library/react';
import { useLocalStorage } from '@/lib/hooks/use-local-storage';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('useLocalStorage', () => {
  beforeEach(() => {
    localStorageMock.clear();
  });

  it('should return initial value when localStorage is empty', () => {
    const { result } = renderHook(() =>
      useLocalStorage('test-key', 'initial-value')
    );

    expect(result.current[0]).toBe('initial-value');
  });

  it('should return stored value from localStorage', () => {
    localStorageMock.setItem('test-key', JSON.stringify('stored-value'));

    const { result } = renderHook(() =>
      useLocalStorage('test-key', 'initial-value')
    );

    expect(result.current[0]).toBe('stored-value');
  });

  it('should update localStorage when value changes', () => {
    const { result } = renderHook(() =>
      useLocalStorage('test-key', 'initial-value')
    );

    act(() => {
      result.current[1]('new-value');
    });

    expect(result.current[0]).toBe('new-value');
    expect(localStorageMock.getItem('test-key')).toBe(
      JSON.stringify('new-value')
    );
  });

  it('should support functional updates', () => {
    const { result } = renderHook(() => useLocalStorage('test-key', 0));

    act(() => {
      result.current[1]((prev) => prev + 1);
    });

    expect(result.current[0]).toBe(1);

    act(() => {
      result.current[1]((prev) => prev + 1);
    });

    expect(result.current[0]).toBe(2);
  });

  it('should remove value from localStorage', () => {
    const { result } = renderHook(() =>
      useLocalStorage('test-key', 'initial-value')
    );

    act(() => {
      result.current[1]('stored-value');
    });

    expect(localStorageMock.getItem('test-key')).toBe(
      JSON.stringify('stored-value')
    );

    act(() => {
      result.current[2]();
    });

    expect(result.current[0]).toBe('initial-value');
    expect(localStorageMock.getItem('test-key')).toBeNull();
  });

  it('should handle complex objects', () => {
    const initialValue = { name: 'test', count: 0 };
    const { result } = renderHook(() =>
      useLocalStorage('test-key', initialValue)
    );

    act(() => {
      result.current[1]({ name: 'updated', count: 5 });
    });

    expect(result.current[0]).toEqual({ name: 'updated', count: 5 });
    expect(localStorageMock.getItem('test-key')).toBe(
      JSON.stringify({ name: 'updated', count: 5 })
    );
  });

  it('should handle arrays', () => {
    const initialValue: number[] = [];
    const { result } = renderHook(() =>
      useLocalStorage('test-key', initialValue)
    );

    act(() => {
      result.current[1]([1, 2, 3]);
    });

    expect(result.current[0]).toEqual([1, 2, 3]);
  });

  it('should sync across tabs via storage event', () => {
    const { result } = renderHook(() =>
      useLocalStorage('test-key', 'initial')
    );

    act(() => {
      result.current[1]('local-update');
    });

    // Simulate storage event from another tab
    act(() => {
      const event = new StorageEvent('storage', {
        key: 'test-key',
        newValue: JSON.stringify('tab-update'),
        oldValue: JSON.stringify('local-update'),
        storageArea: localStorageMock,
      });
      window.dispatchEvent(event);
    });

    expect(result.current[0]).toBe('tab-update');
  });

  it('should handle localStorage errors gracefully', () => {
    // Mock localStorage to throw error
    const originalSetItem = localStorageMock.setItem;
    localStorageMock.setItem = jest.fn(() => {
      throw new Error('QuotaExceededError');
    });

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

    const { result } = renderHook(() =>
      useLocalStorage('test-key', 'initial')
    );

    act(() => {
      result.current[1]('new-value');
    });

    // Should not crash, but value might not be updated
    expect(consoleSpy).toHaveBeenCalled();

    // Restore
    localStorageMock.setItem = originalSetItem;
    consoleSpy.mockRestore();
  });
});

