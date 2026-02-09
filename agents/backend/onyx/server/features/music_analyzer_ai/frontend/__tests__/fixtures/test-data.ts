/**
 * Test Data Fixtures
 * Reusable test data for consistent testing
 */

import { type Track } from '@/lib/api/music-api';

export const mockTrack1: Track = {
  id: 'track-1',
  name: 'Track 1',
  artists: ['Artist 1'],
  album: 'Album 1',
  duration_ms: 200000,
  preview_url: 'https://example.com/preview1.mp3',
  popularity: 80,
  images: [{ url: 'https://example.com/image1.jpg' }],
};

export const mockTrack2: Track = {
  id: 'track-2',
  name: 'Track 2',
  artists: ['Artist 2'],
  album: 'Album 2',
  duration_ms: 180000,
  preview_url: 'https://example.com/preview2.mp3',
  popularity: 75,
  images: [{ url: 'https://example.com/image2.jpg' }],
};

export const mockTrack3: Track = {
  id: 'track-3',
  name: 'Track 3',
  artists: ['Artist 3'],
  album: 'Album 3',
  duration_ms: 220000,
  preview_url: 'https://example.com/preview3.mp3',
  popularity: 85,
  images: [{ url: 'https://example.com/image3.jpg' }],
};

export const mockTrackWithoutPreview: Track = {
  ...mockTrack1,
  preview_url: null,
};

export const mockTrackWithoutImage: Track = {
  ...mockTrack1,
  images: [],
};

export const mockTracks: Track[] = [mockTrack1, mockTrack2, mockTrack3];

export const mockSearchResponse = {
  success: true,
  query: 'test',
  results: mockTracks,
  total: 3,
};

export const mockErrorResponse = {
  success: false,
  error: 'Error message',
};

export const mockApiHealthResponse = {
  status: 'healthy' as const,
  message: 'API is reachable',
  timestamp: Date.now(),
};

export const mockUnhealthyApiResponse = {
  status: 'unhealthy' as const,
  message: 'Connection failed',
  timestamp: Date.now(),
};

