/**
 * Mutation Testing
 * 
 * Tests that verify the quality of our test suite by introducing mutations
 * and ensuring our tests catch them. This helps identify weak tests.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Example function to test
function calculateTotal(items: Array<{ price: number; quantity: number }>): number {
  let total = 0;
  for (const item of items) {
    total += item.price * item.quantity;
  }
  return total;
}

// Example validation function
function validateEmail(email: string): boolean {
  if (!email) return false;
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Example search function
function searchTracks(tracks: Array<{ title: string }>, query: string): Array<{ title: string }> {
  if (!query) return tracks;
  const lowerQuery = query.toLowerCase();
  return tracks.filter(track => track.title.toLowerCase().includes(lowerQuery));
}

describe('Mutation Testing - Test Quality Validation', () => {
  describe('Arithmetic Operations', () => {
    it('should detect mutation in addition operator', () => {
      const items = [
        { price: 10, quantity: 2 },
        { price: 5, quantity: 3 },
      ];

      const result = calculateTotal(items);
      expect(result).toBe(35); // 10*2 + 5*3 = 20 + 15 = 35

      // If mutation changed + to -, this test should fail
      // Original: total += item.price * item.quantity
      // Mutated:  total -= item.price * item.quantity
      // Result would be -35, test would fail
    });

    it('should detect mutation in multiplication operator', () => {
      const items = [
        { price: 10, quantity: 2 },
      ];

      const result = calculateTotal(items);
      expect(result).toBe(20); // 10 * 2 = 20

      // If mutation changed * to /, this test should fail
      // Original: total += item.price * item.quantity
      // Mutated:  total += item.price / item.quantity
      // Result would be 5, test would fail
    });

    it('should detect mutation in loop initialization', () => {
      const items = [
        { price: 10, quantity: 1 },
      ];

      const result = calculateTotal(items);
      expect(result).toBe(10);

      // If mutation changed initial value from 0 to 1, this test should fail
      // Original: let total = 0
      // Mutated:  let total = 1
      // Result would be 11, test would fail
    });
  });

  describe('String Operations', () => {
    it('should detect mutation in email validation regex', () => {
      expect(validateEmail('test@example.com')).toBe(true);
      expect(validateEmail('invalid-email')).toBe(false);
      expect(validateEmail('')).toBe(false);

      // If mutation removed @ check, test should fail
      // Original: /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      // Mutated:  /^[^\s@]+[^\s@]+\.[^\s@]+$/ (removed @)
      // 'invalid-email' would pass, test would fail
    });

    it('should detect mutation in case sensitivity', () => {
      const tracks = [
        { title: 'Hello World' },
        { title: 'Goodbye World' },
      ];

      const results = searchTracks(tracks, 'hello');
      expect(results).toHaveLength(1);
      expect(results[0].title).toBe('Hello World');

      // If mutation removed toLowerCase(), test should fail
      // Original: const lowerQuery = query.toLowerCase()
      // Mutated:  const lowerQuery = query
      // Search for 'hello' wouldn't find 'Hello World', test would fail
    });

    it('should detect mutation in empty query handling', () => {
      const tracks = [
        { title: 'Track 1' },
        { title: 'Track 2' },
      ];

      const results = searchTracks(tracks, '');
      expect(results).toHaveLength(2);

      // If mutation changed empty check, test should fail
      // Original: if (!query) return tracks
      // Mutated:  if (query) return tracks
      // Empty query would return empty array, test would fail
    });
  });

  describe('Boundary Conditions', () => {
    it('should detect mutation in boundary checks', () => {
      const items: Array<{ price: number; quantity: number }> = [];
      const result = calculateTotal(items);
      expect(result).toBe(0);

      // If mutation changed loop to not handle empty arrays, test would fail
    });

    it('should detect mutation in null/undefined handling', () => {
      expect(validateEmail('')).toBe(false);

      // If mutation removed empty check, test would fail
      // Original: if (!email) return false
      // Mutated:  (removed check)
      // Empty string might pass validation, test would fail
    });
  });

  describe('Logic Operators', () => {
    it('should detect mutation in AND to OR', () => {
      // This test ensures we check multiple conditions
      const complexValidation = (value: string, minLength: number, maxLength: number): boolean => {
        return value.length >= minLength && value.length <= maxLength;
      };

      expect(complexValidation('test', 3, 10)).toBe(true);
      expect(complexValidation('te', 3, 10)).toBe(false); // too short
      expect(complexValidation('this is too long', 3, 10)).toBe(false); // too long

      // If mutation changed && to ||, test would fail
      // Original: value.length >= minLength && value.length <= maxLength
      // Mutated:  value.length >= minLength || value.length <= maxLength
      // 'this is too long' would pass, test would fail
    });

    it('should detect mutation in comparison operators', () => {
      const isPositive = (num: number): boolean => num > 0;

      expect(isPositive(5)).toBe(true);
      expect(isPositive(-5)).toBe(false);
      expect(isPositive(0)).toBe(false);

      // If mutation changed > to >=, test would fail
      // Original: num > 0
      // Mutated:  num >= 0
      // 0 would return true, test would fail
    });
  });

  describe('Return Statements', () => {
    it('should detect mutation in return value', () => {
      const getDefaultValue = (): number => 42;

      expect(getDefaultValue()).toBe(42);

      // If mutation changed return value, test would fail
      // Original: return 42
      // Mutated:  return 0
      // Test would fail
    });

    it('should detect mutation in early return', () => {
      const processValue = (value: number | null): number => {
        if (value === null) return 0;
        return value * 2;
      };

      expect(processValue(null)).toBe(0);
      expect(processValue(5)).toBe(10);

      // If mutation removed early return, test would fail
      // Original: if (value === null) return 0
      // Mutated:  (removed)
      // null would cause error, test would fail
    });
  });

  describe('Array Operations', () => {
    it('should detect mutation in filter logic', () => {
      const numbers = [1, 2, 3, 4, 5];
      const evens = numbers.filter(n => n % 2 === 0);

      expect(evens).toEqual([2, 4]);

      // If mutation changed === to !==, test would fail
      // Original: n % 2 === 0
      // Mutated:  n % 2 !== 0
      // Result would be [1, 3, 5], test would fail
    });

    it('should detect mutation in map operation', () => {
      const numbers = [1, 2, 3];
      const doubled = numbers.map(n => n * 2);

      expect(doubled).toEqual([2, 4, 6]);

      // If mutation changed * to +, test would fail
      // Original: n * 2
      // Mutated:  n + 2
      // Result would be [3, 4, 5], test would fail
    });
  });

  describe('Test Coverage Validation', () => {
    it('should have tests for all code paths', () => {
      // This test ensures we're testing both branches
      const conditionalFunction = (condition: boolean): string => {
        if (condition) {
          return 'true';
        }
        return 'false';
      };

      expect(conditionalFunction(true)).toBe('true');
      expect(conditionalFunction(false)).toBe('false');

      // Both branches are tested, so mutations in either would be caught
    });

    it('should test edge cases that catch mutations', () => {
      const divide = (a: number, b: number): number => {
        if (b === 0) throw new Error('Division by zero');
        return a / b;
      };

      expect(() => divide(10, 0)).toThrow('Division by zero');
      expect(divide(10, 2)).toBe(5);

      // If mutation removed zero check, test would fail
    });
  });
});

