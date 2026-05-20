/**
 * Zod validation schemas for music-related data.
 * Provides runtime validation for API requests and responses.
 */

import { z } from 'zod';
import { VALIDATION_LIMITS } from '@/lib/constants';
import {
  nonEmptyStringSchema,
  positiveIntegerSchema,
  stringWithLengthRange,
  arrayWithLengthRange,
  arrayWithMinLength,
} from './common';

/**
 * Search query validation schema.
 */
export const searchQuerySchema = stringWithLengthRange(
  VALIDATION_LIMITS.SEARCH_QUERY_MIN_LENGTH,
  VALIDATION_LIMITS.SEARCH_QUERY_MAX_LENGTH,
  `Query must be at least ${VALIDATION_LIMITS.SEARCH_QUERY_MIN_LENGTH} character`,
  `Query must be at most ${VALIDATION_LIMITS.SEARCH_QUERY_MAX_LENGTH} characters`
);

/**
 * Track ID validation schema.
 */
export const trackIdSchema = nonEmptyStringSchema.refine(
  (val) => val.length >= VALIDATION_LIMITS.TRACK_ID_MIN_LENGTH,
  {
    message: 'Track ID cannot be empty',
  }
);

/**
 * Track IDs array validation schema for comparison.
 */
export const trackIdsSchema = arrayWithLengthRange(
  trackIdSchema,
  VALIDATION_LIMITS.MIN_TRACKS_COMPARE,
  VALIDATION_LIMITS.MAX_TRACKS_COMPARE,
  `At least ${VALIDATION_LIMITS.MIN_TRACKS_COMPARE} tracks are required for comparison`,
  `Maximum ${VALIDATION_LIMITS.MAX_TRACKS_COMPARE} tracks can be compared at once`
);

/**
 * Pagination parameters validation schema.
 */
export const paginationSchema = z.object({
  page: positiveIntegerSchema.default(1),
  limit: z
    .number()
    .int('Limit must be an integer')
    .positive('Limit must be positive')
    .max(100, 'Limit cannot exceed 100')
    .default(20),
});

/**
 * Search request validation schema.
 */
export const searchRequestSchema = z.object({
  query: searchQuerySchema,
  limit: z.number().int().positive().max(50).default(10),
});

/**
 * Analyze track request validation schema.
 */
export const analyzeTrackRequestSchema = z.object({
  trackId: trackIdSchema.optional(),
  trackName: z.string().min(1).optional(),
  includeCoaching: z.boolean().default(true),
}).refine(
  (data) => data.trackId || data.trackName,
  {
    message: 'Either trackId or trackName must be provided',
  }
);

/**
 * Compare tracks request validation schema.
 */
export const compareTracksRequestSchema = z.object({
  trackIds: trackIdsSchema,
  comparisonType: z.enum(['all', 'musical', 'technical']).default('all'),
});

/**
 * User ID validation schema.
 */
export const userIdSchema = nonEmptyStringSchema.refine(
  (val) => val.length >= 1 && val.length <= 100,
  {
    message: 'User ID must be between 1 and 100 characters',
  }
);

/**
 * Playlist name validation schema.
 */
export const playlistNameSchema = stringWithLengthRange(
  1,
  100,
  'Playlist name must be at least 1 character',
  'Playlist name must be at most 100 characters'
);

/**
 * Mood validation schema.
 */
export const moodSchema = z.enum([
  'happy',
  'sad',
  'energetic',
  'calm',
  'romantic',
  'aggressive',
  'melancholic',
  'uplifting',
]);

/**
 * Activity validation schema.
 */
export const activitySchema = z.enum([
  'workout',
  'study',
  'party',
  'relax',
  'commute',
  'sleep',
  'focus',
  'dance',
]);

/**
 * Time of day validation schema.
 */
export const timeOfDaySchema = z.enum(['morning', 'afternoon', 'evening', 'night']);

/**
 * Rating validation schema (1-5 stars).
 */
export const ratingSchema = z
  .number()
  .int('Rating must be an integer')
  .min(1, 'Rating must be at least 1')
  .max(5, 'Rating must be at most 5');

/**
 * Comment content validation schema.
 */
export const commentContentSchema = stringWithLengthRange(
  1,
  1000,
  'Comment must be at least 1 character',
  'Comment must be at most 1000 characters'
);

/**
 * Note content validation schema.
 */
export const noteContentSchema = stringWithLengthRange(
  1,
  5000,
  'Note must be at least 1 character',
  'Note must be at most 5000 characters'
);

/**
 * Tag validation schema.
 */
export const tagSchema = stringWithLengthRange(
  1,
  50,
  'Tag must be at least 1 character',
  'Tag must be at most 50 characters'
);

/**
 * Tags array validation schema.
 */
export const tagsSchema = arrayWithMaxLength(
  tagSchema,
  20,
  'Maximum 20 tags allowed'
);

/**
 * Export format validation schema.
 */
export const exportFormatSchema = z.enum(['json', 'text', 'markdown']);

/**
 * Favorites request validation schema.
 */
export const addToFavoritesRequestSchema = z.object({
  userId: userIdSchema,
  trackId: trackIdSchema,
  trackName: nonEmptyStringSchema,
  artists: arrayWithMinLength(nonEmptyStringSchema, 1, 'At least one artist is required'),
});

/**
 * Recommendations request validation schema.
 */
export const recommendationsRequestSchema = z.object({
  trackId: trackIdSchema,
  limit: z.number().int().positive().max(50).default(20),
  context: z.string().max(200).optional(),
});

/**
 * Mood-based recommendations request schema.
 */
export const moodRecommendationsRequestSchema = z.object({
  mood: moodSchema,
  limit: z.number().int().positive().max(50).default(20),
});

/**
 * Activity-based recommendations request schema.
 */
export const activityRecommendationsRequestSchema = z.object({
  activity: activitySchema,
  limit: z.number().int().positive().max(50).default(20),
});

/**
 * Time-based recommendations request schema.
 */
export const timeRecommendationsRequestSchema = z.object({
  timeOfDay: timeOfDaySchema,
  limit: z.number().int().positive().max(50).default(20),
});

