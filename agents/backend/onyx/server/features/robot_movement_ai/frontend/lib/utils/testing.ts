/**
 * Testing utilities
 */

// Mock function
export function createMockFunction<T extends (...args: any[]) => any>(
  returnValue?: ReturnType<T>
): jest.Mock<ReturnType<T>, Parameters<T>> {
  const mock = jest.fn();
  if (returnValue !== undefined) {
    mock.mockReturnValue(returnValue);
  }
  return mock as jest.Mock<ReturnType<T>, Parameters<T>>;
}

// Wait for condition
export async function waitFor(
  condition: () => boolean,
  timeout: number = 5000,
  interval: number = 100
): Promise<void> {
  const startTime = Date.now();

  return new Promise((resolve, reject) => {
    const check = () => {
      if (condition()) {
        resolve();
      } else if (Date.now() - startTime > timeout) {
        reject(new Error('Timeout waiting for condition'));
      } else {
        setTimeout(check, interval);
      }
    };

    check();
  });
}

// Create test data
export function createTestData<T>(
  factory: (index: number) => T,
  count: number = 10
): T[] {
  return Array.from({ length: count }, (_, i) => factory(i));
}

// Mock API response
export function mockApiResponse<T>(data: T, delay: number = 0): Promise<T> {
  return new Promise((resolve) => {
    setTimeout(() => resolve(data), delay);
  });
}

// Mock API error
export function mockApiError(message: string = 'API Error', delay: number = 0): Promise<never> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), delay);
  });
}

// Assert helpers
export const assert = {
  isDefined: <T>(value: T | null | undefined): asserts value is T => {
    if (value === null || value === undefined) {
      throw new Error('Value is not defined');
    }
  },
  isTrue: (value: boolean, message?: string) => {
    if (!value) {
      throw new Error(message || 'Value is not true');
    }
  },
  isFalse: (value: boolean, message?: string) => {
    if (value) {
      throw new Error(message || 'Value is not false');
    }
  },
  equals: <T>(actual: T, expected: T, message?: string) => {
    if (actual !== expected) {
      throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
  },
  deepEquals: <T>(actual: T, expected: T, message?: string) => {
    if (JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(message || `Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    }
  },
};



