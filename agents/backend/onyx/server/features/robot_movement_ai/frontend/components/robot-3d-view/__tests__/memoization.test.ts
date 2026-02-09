/**
 * Tests for memoization utilities
 * @module robot-3d-view/__tests__/memoization
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { memoize, weakMemoize, memoizeAsync } from '../utils/memoization';

describe('memoize', () => {
  it('should cache function results', () => {
    let callCount = 0;
    const fn = (x: number) => {
      callCount++;
      return x * 2;
    };

    const memoized = memoize(fn);
    expect(memoized(5)).toBe(10);
    expect(memoized(5)).toBe(10);
    expect(callCount).toBe(1);
  });

  it('should respect max size', () => {
    const fn = (x: number) => x;
    const memoized = memoize(fn, undefined, 2);

    memoized(1);
    memoized(2);
    memoized(3); // Should evict 1

    expect(memoized.cache.has(JSON.stringify([1]))).toBe(false);
    expect(memoized.cache.has(JSON.stringify([2]))).toBe(true);
    expect(memoized.cache.has(JSON.stringify([3]))).toBe(true);
  });

  it('should clear cache', () => {
    const fn = (x: number) => x;
    const memoized = memoize(fn);

    memoized(5);
    expect(memoized.cache.size).toBe(1);

    memoized.clear();
    expect(memoized.cache.size).toBe(0);
  });
});

describe('weakMemoize', () => {
  it('should cache using WeakMap', () => {
    let callCount = 0;
    const fn = (obj: { value: number }) => {
      callCount++;
      return obj.value * 2;
    };

    const memoized = weakMemoize(fn);
    const obj = { value: 5 };

    expect(memoized(obj)).toBe(10);
    expect(memoized(obj)).toBe(10);
    expect(callCount).toBe(1);
  });
});

describe('memoizeAsync', () => {
  it('should cache async function results', async () => {
    let callCount = 0;
    const fn = async (x: number) => {
      callCount++;
      return x * 2;
    };

    const memoized = memoizeAsync(fn);
    const result1 = await memoized(5);
    const result2 = await memoized(5);

    expect(result1).toBe(10);
    expect(result2).toBe(10);
    expect(callCount).toBe(1);
  });
});



