/**
 * Favorites API endpoints.
 */

import { musicApiClient } from './client';
import { ValidationError } from '../errors';

/**
 * Get user's favorite tracks.
 * @param userId - User ID (optional)
 * @returns Promise resolving to favorites list
 * @throws {ApiError} If API request fails
 */
export async function getFavorites(userId?: string): Promise<unknown> {
  const response = await musicApiClient.get('/favorites', {
    params: userId ? { user_id: userId } : undefined,
  });

  return response.data;
}

/**
 * Add a track to favorites.
 * @param userId - User ID
 * @param trackId - Spotify track ID
 * @param trackName - Track name
 * @param artists - Array of artist names
 * @returns Promise resolving to success response
 * @throws {ValidationError} If any required parameter is empty
 * @throws {ApiError} If API request fails
 */
export async function addToFavorites(
  userId: string,
  trackId: string,
  trackName: string,
  artists: string[]
): Promise<unknown> {
  if (!userId || userId.trim().length === 0) {
    throw new ValidationError('User ID cannot be empty', 'userId');
  }

  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  if (!trackName || trackName.trim().length === 0) {
    throw new ValidationError('Track name cannot be empty', 'trackName');
  }

  if (!artists || artists.length === 0) {
    throw new ValidationError('Artists array cannot be empty', 'artists');
  }

  const response = await musicApiClient.post(
    '/favorites',
    null,
    {
      params: {
        user_id: userId,
        track_id: trackId,
        track_name: trackName,
        artists: artists.join(','),
      },
    }
  );

  return response.data;
}

/**
 * Remove a track from favorites.
 * @param userId - User ID
 * @param trackId - Spotify track ID
 * @returns Promise resolving to success response
 * @throws {ValidationError} If userId or trackId is empty
 * @throws {ApiError} If API request fails
 */
export async function removeFromFavorites(
  userId: string,
  trackId: string
): Promise<unknown> {
  if (!userId || userId.trim().length === 0) {
    throw new ValidationError('User ID cannot be empty', 'userId');
  }

  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  const response = await musicApiClient.delete(`/favorites/${trackId}`, {
    params: { user_id: userId },
  });

  return response.data;
}

