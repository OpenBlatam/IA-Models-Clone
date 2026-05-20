/**
 * Track-related API endpoints.
 */

import { musicApiClient } from './client';
import {
  TrackSchema,
  TrackSearchResponseSchema,
  MusicAnalysisResponseSchema,
  type Track,
  type TrackSearchResponse,
  type MusicAnalysisResponse,
} from './types';
import { validateOrThrow } from '../utils/validation';
import {
  searchRequestSchema,
  analyzeTrackRequestSchema,
} from '../validations';

/**
 * Search for tracks by query string.
 * @param query - Search query string
 * @param limit - Maximum number of results (default: 10)
 * @returns Promise resolving to track search response
 * @throws {ValidationError} If query is empty
 * @throws {ApiError} If API request fails
 */
export async function searchTracks(
  query: string,
  limit: number = 10
): Promise<TrackSearchResponse> {
  // Validate input with Zod
  const validatedInput = validateOrThrow(
    searchRequestSchema,
    { query, limit },
    'searchRequest'
  );

  const response = await musicApiClient.post<TrackSearchResponse>('/search', {
    query: validatedInput.query,
    limit: validatedInput.limit,
  });

  // Validate response with Zod
  return validateOrThrow(TrackSearchResponseSchema, response.data);
}

/**
 * Analyze a track by ID or name.
 * @param options - Analysis options
 * @param options.trackId - Spotify track ID (optional)
 * @param options.trackName - Track name (optional, required if trackId not provided)
 * @param options.includeCoaching - Whether to include coaching insights (default: true)
 * @returns Promise resolving to music analysis response
 * @throws {ValidationError} If neither trackId nor trackName is provided
 * @throws {ApiError} If API request fails
 */
export async function analyzeTrack(options: {
  trackId?: string;
  trackName?: string;
  includeCoaching?: boolean;
}): Promise<MusicAnalysisResponse> {
  // Validate input with Zod
  const validatedInput = validateOrThrow(
    analyzeTrackRequestSchema,
    options,
    'analyzeTrackRequest'
  );

  const response = await musicApiClient.post<MusicAnalysisResponse>('/analyze', {
    ...(validatedInput.trackId && { track_id: validatedInput.trackId }),
    ...(validatedInput.trackName && { track_name: validatedInput.trackName }),
    include_coaching: validatedInput.includeCoaching,
  });

  // Validate response with Zod
  return validateOrThrow(MusicAnalysisResponseSchema, response.data);
}

/**
 * Get basic track information.
 * @param trackId - Spotify track ID
 * @returns Promise resolving to track info
 * @throws {ValidationError} If trackId is empty
 * @throws {ApiError} If API request fails
 */
export async function getTrackInfo(trackId: string): Promise<unknown> {
  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  const response = await musicApiClient.get(`/track/${trackId}/info`);
  return response.data;
}

/**
 * Get audio features for a track.
 * @param trackId - Spotify track ID
 * @returns Promise resolving to audio features
 * @throws {ValidationError} If trackId is empty
 * @throws {ApiError} If API request fails
 */
export async function getAudioFeatures(trackId: string): Promise<unknown> {
  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  const response = await musicApiClient.get(`/track/${trackId}/audio-features`);
  return response.data;
}

/**
 * Get detailed audio analysis for a track.
 * @param trackId - Spotify track ID
 * @returns Promise resolving to audio analysis
 * @throws {ValidationError} If trackId is empty
 * @throws {ApiError} If API request fails
 */
export async function getAudioAnalysis(trackId: string): Promise<unknown> {
  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  const response = await musicApiClient.get(`/track/${trackId}/audio-analysis`);
  return response.data;
}

