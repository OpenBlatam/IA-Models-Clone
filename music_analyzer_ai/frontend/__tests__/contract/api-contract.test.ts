/**
 * Contract Testing
 * 
 * Tests that verify API contracts are maintained between frontend and backend.
 * Ensures that API responses match expected schemas and that breaking changes are detected.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { z } from 'zod';

// API Contract Schemas
const TrackSchema = z.object({
  id: z.string(),
  title: z.string(),
  artist: z.string(),
  duration: z.number().positive(),
  album: z.string().optional(),
  genre: z.string().optional(),
  year: z.number().int().positive().optional(),
});

const PlaylistSchema = z.object({
  id: z.string(),
  name: z.string(),
  tracks: z.array(TrackSchema),
  createdAt: z.string(),
  updatedAt: z.string().optional(),
});

const SearchResponseSchema = z.object({
  tracks: z.array(TrackSchema),
  total: z.number().int().nonnegative(),
  page: z.number().int().positive(),
  limit: z.number().int().positive(),
});

const ErrorResponseSchema = z.object({
  error: z.string(),
  message: z.string(),
  code: z.string().optional(),
});

describe('API Contract Testing', () => {
  describe('Track API Contract', () => {
    it('should validate track response schema', () => {
      const validTrack = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
        album: 'Test Album',
        genre: 'Rock',
        year: 2023,
      };

      expect(() => TrackSchema.parse(validTrack)).not.toThrow();
    });

    it('should reject track with missing required fields', () => {
      const invalidTrack = {
        id: '123',
        title: 'Test Track',
        // missing artist and duration
      };

      expect(() => TrackSchema.parse(invalidTrack)).toThrow();
    });

    it('should accept track with optional fields missing', () => {
      const minimalTrack = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
      };

      expect(() => TrackSchema.parse(minimalTrack)).not.toThrow();
    });

    it('should reject track with invalid types', () => {
      const invalidTrack = {
        id: 123, // should be string
        title: 'Test Track',
        artist: 'Test Artist',
        duration: '180', // should be number
      };

      expect(() => TrackSchema.parse(invalidTrack)).toThrow();
    });

    it('should validate track duration is positive', () => {
      const invalidTrack = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: -10, // negative duration
      };

      expect(() => TrackSchema.parse(invalidTrack)).toThrow();
    });
  });

  describe('Playlist API Contract', () => {
    it('should validate playlist response schema', () => {
      const validPlaylist = {
        id: 'playlist-1',
        name: 'My Playlist',
        tracks: [
          {
            id: '123',
            title: 'Track 1',
            artist: 'Artist 1',
            duration: 180,
          },
        ],
        createdAt: '2023-01-01T00:00:00Z',
        updatedAt: '2023-01-02T00:00:00Z',
      };

      expect(() => PlaylistSchema.parse(validPlaylist)).not.toThrow();
    });

    it('should validate playlist without updatedAt', () => {
      const playlist = {
        id: 'playlist-1',
        name: 'My Playlist',
        tracks: [],
        createdAt: '2023-01-01T00:00:00Z',
      };

      expect(() => PlaylistSchema.parse(playlist)).not.toThrow();
    });

    it('should reject playlist with invalid tracks', () => {
      const invalidPlaylist = {
        id: 'playlist-1',
        name: 'My Playlist',
        tracks: [
          {
            id: '123',
            // missing required fields
          },
        ],
        createdAt: '2023-01-01T00:00:00Z',
      };

      expect(() => PlaylistSchema.parse(invalidPlaylist)).toThrow();
    });
  });

  describe('Search API Contract', () => {
    it('should validate search response schema', () => {
      const validResponse = {
        tracks: [
          {
            id: '123',
            title: 'Track 1',
            artist: 'Artist 1',
            duration: 180,
          },
        ],
        total: 1,
        page: 1,
        limit: 10,
      };

      expect(() => SearchResponseSchema.parse(validResponse)).not.toThrow();
    });

    it('should validate empty search results', () => {
      const emptyResponse = {
        tracks: [],
        total: 0,
        page: 1,
        limit: 10,
      };

      expect(() => SearchResponseSchema.parse(emptyResponse)).not.toThrow();
    });

    it('should reject search response with negative total', () => {
      const invalidResponse = {
        tracks: [],
        total: -1, // negative total
        page: 1,
        limit: 10,
      };

      expect(() => SearchResponseSchema.parse(invalidResponse)).toThrow();
    });

    it('should reject search response with invalid pagination', () => {
      const invalidResponse = {
        tracks: [],
        total: 0,
        page: 0, // should be positive
        limit: 10,
      };

      expect(() => SearchResponseSchema.parse(invalidResponse)).toThrow();
    });
  });

  describe('Error Response Contract', () => {
    it('should validate error response schema', () => {
      const validError = {
        error: 'ValidationError',
        message: 'Invalid input',
        code: 'INVALID_INPUT',
      };

      expect(() => ErrorResponseSchema.parse(validError)).not.toThrow();
    });

    it('should validate error response without code', () => {
      const error = {
        error: 'ServerError',
        message: 'Internal server error',
      };

      expect(() => ErrorResponseSchema.parse(error)).not.toThrow();
    });

    it('should reject error response with missing required fields', () => {
      const invalidError = {
        error: 'ValidationError',
        // missing message
      };

      expect(() => ErrorResponseSchema.parse(invalidError)).toThrow();
    });
  });

  describe('API Versioning Contract', () => {
    it('should maintain backward compatibility for track schema', () => {
      // Old version format
      const oldTrack = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
      };

      // Should still be valid
      expect(() => TrackSchema.parse(oldTrack)).not.toThrow();
    });

    it('should handle new optional fields gracefully', () => {
      // New version with additional optional field
      const newTrack = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
        bpm: 120, // new optional field
      };

      // Should not break if we ignore unknown fields
      const result = TrackSchema.safeParse(newTrack);
      expect(result.success).toBe(true);
    });
  });

  describe('API Response Consistency', () => {
    it('should ensure consistent response structure across endpoints', () => {
      const trackResponse = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
      };

      const playlistTrack = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
      };

      // Both should validate against the same schema
      expect(() => TrackSchema.parse(trackResponse)).not.toThrow();
      expect(() => TrackSchema.parse(playlistTrack)).not.toThrow();
    });

    it('should ensure error responses are consistent', () => {
      const apiError = {
        error: 'NotFound',
        message: 'Resource not found',
      };

      const validationError = {
        error: 'ValidationError',
        message: 'Invalid input',
      };

      // Both should validate against the same schema
      expect(() => ErrorResponseSchema.parse(apiError)).not.toThrow();
      expect(() => ErrorResponseSchema.parse(validationError)).not.toThrow();
    });
  });

  describe('API Contract Breaking Changes Detection', () => {
    it('should detect when required field is removed', () => {
      const oldResponse = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180,
      };

      // Simulate breaking change: duration removed
      const breakingResponse = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        // duration removed
      };

      expect(() => TrackSchema.parse(oldResponse)).not.toThrow();
      expect(() => TrackSchema.parse(breakingResponse)).toThrow();
    });

    it('should detect when field type changes', () => {
      const oldResponse = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: 180, // number
      };

      // Simulate breaking change: duration becomes string
      const breakingResponse = {
        id: '123',
        title: 'Test Track',
        artist: 'Test Artist',
        duration: '180', // string instead of number
      };

      expect(() => TrackSchema.parse(oldResponse)).not.toThrow();
      expect(() => TrackSchema.parse(breakingResponse)).toThrow();
    });
  });
});

