/**
 * Track comparison API endpoints.
 */

import { musicApiClient } from './client';
import { ComparisonResponseSchema, type ComparisonResponse } from './types';
import { ValidationError } from '../errors';

/**
 * Compare multiple tracks.
 * @param trackIds - Array of Spotify track IDs to compare
 * @returns Promise resolving to comparison response
 * @throws {ValidationError} If trackIds array is empty or has less than 2 tracks
 * @throws {ApiError} If API request fails
 */
export async function compareTracks(
  trackIds: string[]
): Promise<ComparisonResponse> {
  if (!trackIds || trackIds.length < 2) {
    throw new ValidationError(
      'At least 2 track IDs are required for comparison',
      'trackIds'
    );
  }

  if (trackIds.length > 10) {
    throw new ValidationError(
      'Maximum 10 tracks can be compared at once',
      'trackIds'
    );
  }

  // Validate all track IDs are non-empty
  const invalidIds = trackIds.filter((id) => !id || id.trim().length === 0);
  if (invalidIds.length > 0) {
    throw new ValidationError('All track IDs must be non-empty', 'trackIds');
  }

  const response = await musicApiClient.post<ComparisonResponse>('/compare', {
    track_ids: trackIds,
  });

  return ComparisonResponseSchema.parse(response.data);
}

/**
 * Compare tracks using ML analysis.
 * @param trackIds - Array of Spotify track IDs to compare
 * @param comparisonType - Type of comparison (default: 'all')
 * @returns Promise resolving to ML comparison results
 * @throws {ValidationError} If trackIds array is invalid
 * @throws {ApiError} If API request fails
 */
export async function compareTracksML(
  trackIds: string[],
  comparisonType: string = 'all'
): Promise<unknown> {
  if (!trackIds || trackIds.length < 2) {
    throw new ValidationError(
      'At least 2 track IDs are required for comparison',
      'trackIds'
    );
  }

  const response = await musicApiClient.post(
    '/ml/compare-tracks',
    {
      track_ids: trackIds,
    },
    {
      params: { comparison_type: comparisonType },
    }
  );

  return response.data;
}

