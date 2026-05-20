/**
 * Recommendations API endpoints.
 */

import { musicApiClient } from './client';
import { ValidationError } from '../errors';

/**
 * Get track recommendations based on a track.
 * @param trackId - Spotify track ID
 * @param limit - Maximum number of recommendations (default: 20)
 * @returns Promise resolving to recommendations
 * @throws {ValidationError} If trackId is empty or limit is invalid
 * @throws {ApiError} If API request fails
 */
export async function getRecommendations(
  trackId: string,
  limit: number = 20
): Promise<unknown> {
  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  if (limit < 1 || limit > 50) {
    throw new ValidationError('Limit must be between 1 and 50', 'limit');
  }

  const response = await musicApiClient.get(`/track/${trackId}/recommendations`, {
    params: { limit },
  });

  return response.data;
}

/**
 * Get contextual recommendations based on track and context.
 * @param trackId - Spotify track ID
 * @param context - Context string (optional)
 * @param limit - Maximum number of recommendations (default: 20)
 * @returns Promise resolving to contextual recommendations
 * @throws {ValidationError} If trackId is empty
 * @throws {ApiError} If API request fails
 */
export async function getContextualRecommendations(
  trackId: string,
  context?: string,
  limit: number = 20
): Promise<unknown> {
  if (!trackId || trackId.trim().length === 0) {
    throw new ValidationError('Track ID cannot be empty', 'trackId');
  }

  const response = await musicApiClient.post('/recommendations/contextual', {
    track_id: trackId,
    context,
    limit,
  });

  return response.data;
}

/**
 * Get recommendations based on mood.
 * @param mood - Mood string
 * @param limit - Maximum number of recommendations (default: 20)
 * @returns Promise resolving to mood-based recommendations
 * @throws {ValidationError} If mood is empty
 * @throws {ApiError} If API request fails
 */
export async function getRecommendationsByMood(
  mood: string,
  limit: number = 20
): Promise<unknown> {
  if (!mood || mood.trim().length === 0) {
    throw new ValidationError('Mood cannot be empty', 'mood');
  }

  const response = await musicApiClient.post('/recommendations/mood', {
    mood,
    limit,
  });

  return response.data;
}

/**
 * Get recommendations based on activity.
 * @param activity - Activity string
 * @param limit - Maximum number of recommendations (default: 20)
 * @returns Promise resolving to activity-based recommendations
 * @throws {ValidationError} If activity is empty
 * @throws {ApiError} If API request fails
 */
export async function getRecommendationsByActivity(
  activity: string,
  limit: number = 20
): Promise<unknown> {
  if (!activity || activity.trim().length === 0) {
    throw new ValidationError('Activity cannot be empty', 'activity');
  }

  const response = await musicApiClient.post('/recommendations/activity', {
    activity,
    limit,
  });

  return response.data;
}

/**
 * Get recommendations based on time of day.
 * @param timeOfDay - Time of day string (e.g., 'morning', 'afternoon', 'evening', 'night')
 * @param limit - Maximum number of recommendations (default: 20)
 * @returns Promise resolving to time-based recommendations
 * @throws {ValidationError} If timeOfDay is empty
 * @throws {ApiError} If API request fails
 */
export async function getRecommendationsByTimeOfDay(
  timeOfDay: string,
  limit: number = 20
): Promise<unknown> {
  if (!timeOfDay || timeOfDay.trim().length === 0) {
    throw new ValidationError('Time of day cannot be empty', 'timeOfDay');
  }

  const response = await musicApiClient.post('/recommendations/time-of-day', {
    time_of_day: timeOfDay,
    limit,
  });

  return response.data;
}

