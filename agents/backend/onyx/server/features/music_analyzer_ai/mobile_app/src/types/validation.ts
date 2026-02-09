import { z } from 'zod';

export const trackSearchSchema = z.object({
  query: z.string().min(1, 'Search query is required'),
  limit: z.number().int().positive().max(50).optional(),
});

export const analyzeTrackSchema = z.object({
  track_id: z.string().optional(),
  track_name: z.string().optional(),
  include_coaching: z.boolean().optional(),
}).refine(
  (data) => data.track_id || data.track_name,
  {
    message: 'Either track_id or track_name must be provided',
    path: ['track_id'],
  }
);

export const compareTracksSchema = z.object({
  track_ids: z
    .array(z.string().min(1))
    .min(2, 'At least 2 tracks are required')
    .max(5, 'Maximum 5 tracks can be compared'),
});

export const contextualRecommendationSchema = z.object({
  track_id: z.string().min(1, 'Track ID is required'),
  context: z.enum(['morning', 'afternoon', 'evening', 'night']),
});

export const activityRecommendationSchema = z.object({
  track_id: z.string().min(1, 'Track ID is required'),
  activity: z.enum([
    'workout',
    'study',
    'relax',
    'party',
    'commute',
    'sleep',
  ]),
});

export const moodRecommendationSchema = z.object({
  track_id: z.string().min(1, 'Track ID is required'),
  mood: z.enum(['happy', 'sad', 'energetic', 'calm', 'focused', 'romantic']),
});

export const discoverySchema = z.object({
  limit: z.number().int().positive().max(50).optional().default(20),
  genre: z.string().optional(),
});

export const artistComparisonSchema = z.object({
  artist_ids: z
    .array(z.string().min(1))
    .min(2, 'At least 2 artists are required')
    .max(5, 'Maximum 5 artists can be compared'),
});

export interface TrackSearchInput {
  query: string;
  limit?: number;
}

export interface AnalyzeTrackInput {
  track_id?: string;
  track_name?: string;
  include_coaching?: boolean;
}

export interface CompareTracksInput {
  track_ids: string[];
}

export interface ContextualRecommendationInput {
  track_id: string;
  context: 'morning' | 'afternoon' | 'evening' | 'night';
}

export interface ActivityRecommendationInput {
  track_id: string;
  activity:
    | 'workout'
    | 'study'
    | 'relax'
    | 'party'
    | 'commute'
    | 'sleep';
}

export interface MoodRecommendationInput {
  track_id: string;
  mood: 'happy' | 'sad' | 'energetic' | 'calm' | 'focused' | 'romantic';
}

export interface DiscoveryInput {
  limit?: number;
  genre?: string;
}

export interface ArtistComparisonInput {
  artist_ids: string[];
}

