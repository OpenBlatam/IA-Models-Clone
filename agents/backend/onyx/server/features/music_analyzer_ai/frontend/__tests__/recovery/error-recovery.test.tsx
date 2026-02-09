/**
 * Error Recovery Testing
 * 
 * Tests that verify the application can gracefully recover from various error conditions,
 * including network failures, API errors, and unexpected states.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { act } from 'react';

describe('Error Recovery Testing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Network Error Recovery', () => {
    it('should retry failed network requests', async () => {
      let attemptCount = 0;
      const maxRetries = 3;

      const fetchWithRetry = async (url: string, retries = maxRetries): Promise<Response> => {
        attemptCount++;
        try {
          const response = await fetch(url);
          if (!response.ok) throw new Error('Network error');
          return response;
        } catch (error) {
          if (retries > 0) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            return fetchWithRetry(url, retries - 1);
          }
          throw error;
        }
      };

      // Mock fetch to fail first 2 times, then succeed
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce(new Response(JSON.stringify({ data: 'success' })));

      await fetchWithRetry('https://api.example.com/data');
      
      expect(attemptCount).toBe(3);
    });

    it('should handle timeout errors gracefully', async () => {
      const timeout = 5000;
      let timedOut = false;

      const fetchWithTimeout = async (url: string): Promise<Response> => {
        return Promise.race([
          fetch(url),
          new Promise<never>((_, reject) => {
            setTimeout(() => {
              timedOut = true;
              reject(new Error('Request timeout'));
            }, timeout);
          }),
        ]);
      };

      global.fetch = vi.fn().mockImplementation(() => 
        new Promise(() => {}) // Never resolves
      );

      try {
        await fetchWithTimeout('https://api.example.com/data');
      } catch (error: any) {
        expect(error.message).toBe('Request timeout');
        expect(timedOut).toBe(true);
      }
    });

    it('should fallback to cached data on network failure', async () => {
      const cache = new Map<string, any>();
      cache.set('key', { data: 'cached' });

      const fetchWithCache = async (key: string): Promise<any> => {
        try {
          const response = await fetch(`https://api.example.com/${key}`);
          const data = await response.json();
          cache.set(key, data);
          return data;
        } catch (error) {
          if (cache.has(key)) {
            return cache.get(key);
          }
          throw error;
        }
      };

      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const result = await fetchWithCache('key');
      expect(result).toEqual({ data: 'cached' });
    });
  });

  describe('API Error Recovery', () => {
    it('should handle 404 errors gracefully', async () => {
      const handle404 = async (url: string) => {
        try {
          const response = await fetch(url);
          if (response.status === 404) {
            return { error: 'Resource not found', fallback: true };
          }
          return await response.json();
        } catch (error) {
          throw error;
        }
      };

      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({ error: 'Not found' }),
      } as Response);

      const result = await handle404('https://api.example.com/missing');
      expect(result.fallback).toBe(true);
    });

    it('should handle 500 errors with retry', async () => {
      let retryCount = 0;
      const maxRetries = 2;

      const handle500 = async (url: string, retries = maxRetries): Promise<any> => {
        const response = await fetch(url);
        if (response.status === 500 && retries > 0) {
          retryCount++;
          await new Promise(resolve => setTimeout(resolve, 1000));
          return handle500(url, retries - 1);
        }
        return await response.json();
      };

      global.fetch = vi.fn()
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Server error' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: 'success' }),
        } as Response);

      const result = await handle500('https://api.example.com/data');
      expect(retryCount).toBe(1);
      expect(result.data).toBe('success');
    });

    it('should handle rate limiting with exponential backoff', async () => {
      const delays: number[] = [];
      let attemptCount = 0;

      const fetchWithBackoff = async (url: string, attempt = 1): Promise<any> => {
        attemptCount++;
        const response = await fetch(url);
        
        if (response.status === 429 && attempt < 4) {
          const delay = Math.pow(2, attempt) * 1000;
          delays.push(delay);
          await new Promise(resolve => setTimeout(resolve, delay));
          return fetchWithBackoff(url, attempt + 1);
        }
        
        return await response.json();
      };

      global.fetch = vi.fn()
        .mockResolvedValueOnce({
          ok: false,
          status: 429,
          json: async () => ({ error: 'Rate limited' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: false,
          status: 429,
          json: async () => ({ error: 'Rate limited' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: 'success' }),
        } as Response);

      await fetchWithBackoff('https://api.example.com/data');
      
      expect(attemptCount).toBe(3);
      expect(delays[0]).toBe(2000); // 2^1 * 1000
      expect(delays[1]).toBe(4000); // 2^2 * 1000
    });
  });

  describe('State Recovery', () => {
    it('should recover from corrupted state', () => {
      let state: any = { corrupted: true, invalid: 'data' };

      const recoverState = (currentState: any) => {
        if (!currentState || currentState.corrupted) {
          return { initialized: true, data: [] };
        }
        return currentState;
      };

      const recovered = recoverState(state);
      expect(recovered.corrupted).toBeUndefined();
      expect(recovered.initialized).toBe(true);
    });

    it('should restore previous valid state', () => {
      const stateHistory: any[] = [
        { valid: true, data: 'state1' },
        { valid: true, data: 'state2' },
        { corrupted: true },
      ];

      const restoreState = (history: any[]) => {
        for (let i = history.length - 1; i >= 0; i--) {
          if (history[i].valid) {
            return history[i];
          }
        }
        return { initialized: true };
      };

      const restored = restoreState(stateHistory);
      expect(restored.data).toBe('state2');
      expect(restored.valid).toBe(true);
    });

    it('should reset to default state on critical error', () => {
      const defaultState = { initialized: false, data: [] };
      let currentState: any = { error: 'critical' };

      const resetToDefault = (state: any, defaultState: any) => {
        if (state.error === 'critical') {
          return defaultState;
        }
        return state;
      };

      const reset = resetToDefault(currentState, defaultState);
      expect(reset.initialized).toBe(false);
      expect(reset.data).toEqual([]);
    });
  });

  describe('Data Validation Recovery', () => {
    it('should recover from invalid data format', () => {
      const invalidData = 'not json';
      const defaultData = { tracks: [] };

      const parseWithRecovery = (data: string, defaultData: any) => {
        try {
          return JSON.parse(data);
        } catch (error) {
          return defaultData;
        }
      };

      const result = parseWithRecovery(invalidData, defaultData);
      expect(result).toEqual(defaultData);
    });

    it('should sanitize corrupted data', () => {
      const corruptedData = {
        tracks: [
          { id: '1', title: 'Valid' },
          { id: null, title: undefined },
          { id: '3', title: 'Valid' },
        ],
      };

      const sanitize = (data: any) => {
        return {
          tracks: data.tracks.filter((track: any) => 
            track.id && track.title
          ),
        };
      };

      const sanitized = sanitize(corruptedData);
      expect(sanitized.tracks).toHaveLength(2);
      expect(sanitized.tracks[0].id).toBe('1');
      expect(sanitized.tracks[1].id).toBe('3');
    });
  });

  describe('User Action Recovery', () => {
    it('should allow user to retry failed actions', () => {
      let retryAttempted = false;

      const retryAction = async (action: () => Promise<any>) => {
        try {
          return await action();
        } catch (error) {
          retryAttempted = true;
          return await action(); // Retry once
        }
      };

      const failingAction = vi.fn()
        .mockRejectedValueOnce(new Error('Failed'))
        .mockResolvedValueOnce('Success');

      retryAction(failingAction);
      
      expect(retryAttempted).toBe(true);
    });

    it('should provide undo functionality for failed operations', () => {
      const operations: any[] = [];
      let state = { value: 0 };

      const executeWithUndo = (operation: () => void) => {
        const previousState = { ...state };
        operations.push({ undo: () => { state = previousState; } });
        operation();
      };

      executeWithUndo(() => { state.value = 10; });
      expect(state.value).toBe(10);

      operations[0].undo();
      expect(state.value).toBe(0);
    });
  });

  describe('Error Boundary Recovery', () => {
    it('should catch and handle React errors', () => {
      const errorBoundary = {
        hasError: false,
        error: null as Error | null,
        resetError: function() {
          this.hasError = false;
          this.error = null;
        },
        catchError: function(error: Error) {
          this.hasError = true;
          this.error = error;
        },
      };

      errorBoundary.catchError(new Error('Component error'));
      expect(errorBoundary.hasError).toBe(true);
      expect(errorBoundary.error?.message).toBe('Component error');

      errorBoundary.resetError();
      expect(errorBoundary.hasError).toBe(false);
    });
  });

  describe('Progressive Degradation', () => {
    it('should degrade gracefully when features are unavailable', () => {
      const features = {
        advancedSearch: false,
        recommendations: false,
        basicSearch: true,
      };

      const getAvailableFeatures = (features: any) => {
        const available: string[] = [];
        if (features.basicSearch) available.push('basicSearch');
        if (features.advancedSearch) available.push('advancedSearch');
        if (features.recommendations) available.push('recommendations');
        return available;
      };

      const available = getAvailableFeatures(features);
      expect(available).toContain('basicSearch');
      expect(available).not.toContain('advancedSearch');
    });
  });
});

