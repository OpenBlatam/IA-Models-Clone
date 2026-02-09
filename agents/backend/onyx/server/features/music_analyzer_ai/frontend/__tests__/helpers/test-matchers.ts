/**
 * Custom Test Matchers
 * Custom Jest matchers for better test assertions
 */

import { expect } from '@jest/globals';

/**
 * Custom matcher to check if a value is a valid Track
 */
expect.extend({
  toBeValidTrack(received: any) {
    const pass =
      received &&
      typeof received.id === 'string' &&
      typeof received.name === 'string' &&
      Array.isArray(received.artists) &&
      typeof received.duration_ms === 'number';

    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid track`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid track`,
        pass: false,
      };
    }
  },
});

/**
 * Custom matcher to check if a value is a valid API response
 */
expect.extend({
  toBeValidApiResponse(received: any) {
    const pass =
      received &&
      typeof received.success === 'boolean' &&
      (received.data !== undefined || received.error !== undefined);

    if (pass) {
      return {
        message: () => `expected ${received} not to be a valid API response`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be a valid API response`,
        pass: false,
      };
    }
  },
});

// Type declarations for TypeScript
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeValidTrack(): R;
      toBeValidApiResponse(): R;
    }
  }
}

