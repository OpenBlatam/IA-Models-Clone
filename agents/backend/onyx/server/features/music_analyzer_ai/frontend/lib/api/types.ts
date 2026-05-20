/**
 * Type definitions for the Music Analyzer API.
 * All types are properly typed to avoid using 'any'.
 */

import { z } from 'zod';

/**
 * Zod schema for Track validation.
 */
export const TrackSchema = z.object({
  id: z.string(),
  name: z.string(),
  artists: z.array(z.string()),
  album: z.string(),
  duration_ms: z.number().int().positive(),
  preview_url: z.string().url().optional(),
  popularity: z.number().int().min(0).max(100),
  external_urls: z
    .object({
      spotify: z.string().url().optional(),
    })
    .optional(),
  images: z
    .array(
      z.object({
        url: z.string().url(),
        height: z.number().int().positive().optional(),
        width: z.number().int().positive().optional(),
      })
    )
    .optional(),
});

/**
 * Track type inferred from Zod schema.
 */
export type Track = z.infer<typeof TrackSchema>;

/**
 * Track search response schema.
 */
export const TrackSearchResponseSchema = z.object({
  success: z.boolean(),
  query: z.string(),
  results: z.array(TrackSchema),
  total: z.number().int().nonnegative(),
});

/**
 * Track search response type.
 */
export type TrackSearchResponse = z.infer<typeof TrackSearchResponseSchema>;

/**
 * Musical analysis schema.
 */
export const MusicalAnalysisSchema = z.object({
  key_signature: z.string(),
  root_note: z.string(),
  mode: z.string(),
  tempo: z.object({
    bpm: z.number().positive(),
    category: z.string(),
  }),
  time_signature: z.string(),
  scale: z.object({
    name: z.string(),
    notes: z.array(z.string()),
  }),
});

/**
 * Musical analysis type.
 */
export type MusicalAnalysis = z.infer<typeof MusicalAnalysisSchema>;

/**
 * Technical analysis feature schema.
 */
export const TechnicalFeatureSchema = z.object({
  value: z.number().min(0).max(1),
  description: z.string(),
});

/**
 * Technical analysis schema.
 */
export const TechnicalAnalysisSchema = z.object({
  energy: TechnicalFeatureSchema,
  danceability: TechnicalFeatureSchema,
  valence: TechnicalFeatureSchema,
  acousticness: TechnicalFeatureSchema,
  instrumentalness: TechnicalFeatureSchema,
  liveness: TechnicalFeatureSchema,
  loudness: z.object({
    value: z.number(),
    description: z.string(),
  }),
});

/**
 * Technical analysis type.
 */
export type TechnicalAnalysis = z.infer<typeof TechnicalAnalysisSchema>;

/**
 * Music analysis response schema.
 */
export const MusicAnalysisResponseSchema = z.object({
  success: z.boolean(),
  track_basic_info: z.object({
    name: z.string(),
    artists: z.array(z.string()),
    album: z.string(),
    duration_seconds: z.number().positive(),
  }),
  musical_analysis: MusicalAnalysisSchema,
  technical_analysis: TechnicalAnalysisSchema,
  composition_analysis: z.unknown().optional(),
  educational_insights: z.unknown().optional(),
  coaching: z.unknown().optional(),
  analysis_id: z.string().optional(),
  lyrics: z.string().optional(),
});

/**
 * Music analysis response type.
 */
export type MusicAnalysisResponse = z.infer<typeof MusicAnalysisResponseSchema>;

/**
 * Comparison response schema.
 */
export const ComparisonResponseSchema = z.object({
  success: z.boolean(),
  comparison: z.object({
    key_signatures: z.object({
      all_same: z.boolean(),
      keys: z.array(z.string()),
      most_common: z.string(),
    }),
    tempos: z.object({
      average: z.number(),
      min: z.number(),
      max: z.number(),
      range: z.number(),
    }),
  }),
  similarities: z.array(z.unknown()),
  differences: z.array(z.unknown()),
  recommendations: z.array(z.unknown()),
});

/**
 * Comparison response type.
 */
export type ComparisonResponse = z.infer<typeof ComparisonResponseSchema>;

/**
 * Generic API response wrapper.
 */
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}


