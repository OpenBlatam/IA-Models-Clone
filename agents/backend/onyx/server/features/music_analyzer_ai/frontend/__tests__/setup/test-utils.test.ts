/**
 * Tests for Test Utilities
 * Ensures test helpers work correctly
 */

import {
  createTestQueryClient,
  createMockTrack,
  createMockTracks,
  createMockApiResponse,
  createMockPaginatedResponse,
  createMockError,
  createMockNetworkError,
  wait,
} from './test-utils';

describe('Test Utilities', () => {
  describe('createTestQueryClient', () => {
    it('should create a QueryClient instance', () => {
      const client = createTestQueryClient();
      expect(client).toBeDefined();
      expect(client.getDefaultOptions().queries?.retry).toBe(false);
    });
  });

  describe('createMockTrack', () => {
    it('should create a mock track with defaults', () => {
      const track = createMockTrack();
      expect(track.id).toBe('track-123');
      expect(track.name).toBe('Test Track');
      expect(track.artists).toEqual(['Test Artist']);
    });

    it('should allow overrides', () => {
      const track = createMockTrack({ name: 'Custom Track', id: 'custom-1' });
      expect(track.name).toBe('Custom Track');
      expect(track.id).toBe('custom-1');
      expect(track.artists).toEqual(['Test Artist']); // Default preserved
    });
  });

  describe('createMockTracks', () => {
    it('should create multiple tracks', () => {
      const tracks = createMockTracks(5);
      expect(tracks).toHaveLength(5);
      expect(tracks[0].id).toBe('track-1');
      expect(tracks[4].id).toBe('track-5');
    });

    it('should create unique tracks', () => {
      const tracks = createMockTracks(3);
      expect(tracks[0].name).toBe('Track 1');
      expect(tracks[1].name).toBe('Track 2');
      expect(tracks[2].name).toBe('Track 3');
    });
  });

  describe('createMockApiResponse', () => {
    it('should create success response', () => {
      const response = createMockApiResponse({ data: 'test' });
      expect(response.success).toBe(true);
      expect(response.data).toEqual({ data: 'test' });
    });

    it('should create error response', () => {
      const response = createMockApiResponse(null, false);
      expect(response.success).toBe(false);
      expect(response.error).toBe('Error message');
    });
  });

  describe('createMockPaginatedResponse', () => {
    it('should create paginated response', () => {
      const items = ['item1', 'item2', 'item3'];
      const response = createMockPaginatedResponse(items, 1, 2);
      expect(response.data).toEqual(items);
      expect(response.meta.page).toBe(1);
      expect(response.meta.limit).toBe(2);
      expect(response.meta.total).toBe(3);
    });

    it('should calculate totalPages correctly', () => {
      const items = Array.from({ length: 25 }, (_, i) => `item${i}`);
      const response = createMockPaginatedResponse(items, 1, 10);
      expect(response.meta.totalPages).toBe(3);
    });
  });

  describe('createMockError', () => {
    it('should create error with default message', () => {
      const error = createMockError();
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('Test error');
    });

    it('should create error with custom message', () => {
      const error = createMockError('Custom error');
      expect(error.message).toBe('Custom error');
    });
  });

  describe('createMockNetworkError', () => {
    it('should create network error', () => {
      const error = createMockNetworkError();
      expect(error).toBeInstanceOf(Error);
      expect(error.message).toBe('Network Error');
      expect((error as any).code).toBe('NETWORK_ERROR');
    });
  });

  describe('wait', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should wait for specified time', async () => {
      const promise = wait(1000);
      jest.advanceTimersByTime(1000);
      await promise;
      // Should complete without error
      expect(true).toBe(true);
    });
  });
});

