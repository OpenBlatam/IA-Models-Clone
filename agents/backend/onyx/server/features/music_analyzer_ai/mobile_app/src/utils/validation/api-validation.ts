import { z } from 'zod';
import type { Track, TrackSearchResponse, TrackAnalysis } from '../../types/api';

export const TrackSchema: z.ZodType<Track> = z.object({
  id: z.string(),
  name: z.string(),
  artists: z.array(z.string()),
  album: z.string(),
  duration_ms: z.number(),
  preview_url: z.string().nullable(),
  external_urls: z.object({
    spotify: z.string(),
  }),
  popularity: z.number().min(0).max(100),
});

export const TrackSearchResponseSchema: z.ZodType<TrackSearchResponse> = z.object({
  success: z.boolean(),
  query: z.string(),
  results: z.array(TrackSchema),
  total: z.number(),
});

export const TrackAnalysisSchema: z.ZodType<TrackAnalysis> = z.object({
  success: z.boolean(),
  track_basic_info: z.object({
    name: z.string(),
    artists: z.array(z.string()),
    album: z.string(),
    duration_seconds: z.number(),
  }),
  musical_analysis: z.object({
    key_signature: z.string(),
    root_note: z.string(),
    mode: z.string(),
    tempo: z.object({
      bpm: z.number(),
      category: z.string(),
    }),
    time_signature: z.string(),
    scale: z.object({
      name: z.string(),
      notes: z.array(z.string()),
    }),
  }),
  technical_analysis: z.object({
    energy: z.number().min(0).max(1),
    danceability: z.number().min(0).max(1),
    valence: z.number().min(0).max(1),
    acousticness: z.number().min(0).max(1),
    instrumentalness: z.number().min(0).max(1),
    liveness: z.number().min(0).max(1),
    speechiness: z.number().min(0).max(1),
    loudness: z.number().optional(),
    tempo: z.number().optional(),
    key: z.number().optional(),
    mode: z.number().optional(),
    time_signature: z.number().optional(),
  }),
  composition_analysis: z.object({
    structure: z.object({
      sections: z.array(
        z.object({
          start: z.number(),
          duration: z.number(),
          confidence: z.number(),
          type: z.string(),
        })
      ),
    }),
    harmony: z.object({
      chords: z.array(z.string()),
      progressions: z.array(z.string()),
    }),
  }),
  performance_analysis: z.object({
    loudness: z.number(),
    dynamics: z.string(),
    intensity: z.string(),
  }),
  educational_insights: z.array(
    z.object({
      concept: z.string(),
      explanation: z.string(),
      examples: z.array(z.string()),
    })
  ),
  coaching: z
    .object({
      learning_path: z.array(
        z.object({
          step: z.number(),
          title: z.string(),
          description: z.string(),
          exercises: z.array(z.string()),
        })
      ),
      practice_exercises: z.array(
        z.object({
          name: z.string(),
          description: z.string(),
          difficulty: z.string(),
          duration_minutes: z.number(),
        })
      ),
      tips: z.array(z.string()),
      recommendations: z.array(z.string()),
    })
    .optional(),
});

export function validateTrack(data: unknown): Track {
  return TrackSchema.parse(data);
}

export function validateTrackSearchResponse(data: unknown): TrackSearchResponse {
  return TrackSearchResponseSchema.parse(data);
}

export function validateTrackAnalysis(data: unknown): TrackAnalysis {
  return TrackAnalysisSchema.parse(data);
}

export function safeValidateTrack(data: unknown): Track | null {
  try {
    return TrackSchema.parse(data);
  } catch {
    return null;
  }
}

export function safeValidateTrackSearchResponse(
  data: unknown
): TrackSearchResponse | null {
  try {
    return TrackSearchResponseSchema.parse(data);
  } catch {
    return null;
  }
}

export function safeValidateTrackAnalysis(data: unknown): TrackAnalysis | null {
  try {
    return TrackAnalysisSchema.parse(data);
  } catch {
    return null;
  }
}


