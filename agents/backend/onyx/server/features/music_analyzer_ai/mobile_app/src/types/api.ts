export interface Track {
  id: string;
  name: string;
  artists: string[];
  album: string;
  duration_ms: number;
  preview_url: string | null;
  external_urls: {
    spotify: string;
  };
  popularity: number;
}

export interface TrackSearchRequest {
  query: string;
  limit?: number;
}

export interface TrackSearchResponse {
  success: boolean;
  query: string;
  results: Track[];
  total: number;
}

export interface TempoAnalysis {
  bpm: number;
  category: string;
}

export interface Scale {
  name: string;
  notes: string[];
}

export interface MusicalAnalysis {
  key_signature: string;
  root_note: string;
  mode: string;
  tempo: TempoAnalysis;
  time_signature: string;
  scale: Scale;
}

export interface TechnicalAnalysis {
  energy: number;
  danceability: number;
  valence: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  speechiness: number;
  loudness?: number;
  tempo?: number;
  key?: number;
  mode?: number;
  time_signature?: number;
}

export interface CompositionAnalysis {
  structure: {
    sections: Array<{
      start: number;
      duration: number;
      confidence: number;
      type: string;
    }>;
  };
  harmony: {
    chords: string[];
    progressions: string[];
  };
}

export interface PerformanceAnalysis {
  loudness: number;
  dynamics: string;
  intensity: string;
}

export interface EducationalInsight {
  concept: string;
  explanation: string;
  examples: string[];
}

export interface TrackAnalysis {
  success: boolean;
  track_basic_info: {
    name: string;
    artists: string[];
    album: string;
    duration_seconds: number;
  };
  musical_analysis: MusicalAnalysis;
  technical_analysis: TechnicalAnalysis;
  composition_analysis: CompositionAnalysis;
  performance_analysis: PerformanceAnalysis;
  educational_insights: EducationalInsight[];
  coaching?: CoachingAnalysis;
}

export interface CoachingAnalysis {
  learning_path: Array<{
    step: number;
    title: string;
    description: string;
    exercises: string[];
  }>;
  practice_exercises: Array<{
    name: string;
    description: string;
    difficulty: string;
    duration_minutes: number;
  }>;
  tips: string[];
  recommendations: string[];
}

export interface TrackComparison {
  success: boolean;
  tracks: Array<{
    track_id: string;
    track_name: string;
    artists: string[];
    similarities: {
      key: number;
      tempo: number;
      energy: number;
      overall: number;
    };
    differences: string[];
  }>;
  summary: string;
}

export interface Recommendation {
  track_id: string;
  track_name: string;
  artists: string[];
  similarity_score: number;
  reasons: string[];
}

export interface RecommendationsResponse {
  success: boolean;
  recommendations: Recommendation[];
  total: number;
}

export interface HistoryEntry {
  id: string;
  track_id: string;
  track_name: string;
  artists: string[];
  analyzed_at: string;
  analysis: TrackAnalysis;
}

export interface ApiError {
  detail: string;
  status_code: number;
}

export interface HealthCheck {
  status: string;
  spotify_connection: string;
  error?: string;
}

export interface HistoryResponse {
  success: boolean;
  history: HistoryEntry[];
  total: number;
}

export interface HistoryStats {
  success: boolean;
  stats: {
    total_analyses: number;
    unique_tracks: number;
    most_analyzed_track?: string;
    average_per_day?: number;
  };
}

export interface ExportResponse {
  success: boolean;
  format: string;
  content: string;
  content_type: string;
  track_id: string;
}

export interface ContextualRecommendationRequest {
  context: string;
  limit?: number;
}

export interface DiscoveryRequest {
  artist_id?: string;
  mood?: string;
  limit?: number;
}

export interface ArtistComparisonRequest {
  artist_ids: string[];
}

export interface TrendsRequest {
  time_range?: 'short_term' | 'medium_term' | 'long_term';
  limit?: number;
}

export interface AudioFeatures {
  danceability: number;
  energy: number;
  key: number;
  loudness: number;
  mode: number;
  speechiness: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  valence: number;
  tempo: number;
  time_signature: number;
  duration_ms: number;
}

export interface AudioAnalysis {
  bars: Array<{
    start: number;
    duration: number;
    confidence: number;
  }>;
  beats: Array<{
    start: number;
    duration: number;
    confidence: number;
  }>;
  sections: Array<{
    start: number;
    duration: number;
    confidence: number;
    loudness: number;
    tempo: number;
    key: number;
    mode: number;
    time_signature: number;
  }>;
  segments: Array<{
    start: number;
    duration: number;
    confidence: number;
    loudness_start: number;
    loudness_max: number;
    pitches: number[];
    timbre: number[];
  }>;
  tatums: Array<{
    start: number;
    duration: number;
    confidence: number;
  }>;
}

export interface ArtistComparison {
  success: boolean;
  artists: Array<{
    artist_id: string;
    artist_name: string;
    genres: string[];
    popularity: number;
    followers: number;
    albums_count: number;
    tracks_count: number;
  }>;
  similarities: {
    genres: string[];
    common_tracks: number;
    style_overlap: number;
  };
  differences: Array<{
    aspect: string;
    description: string;
  }>;
  summary: string;
}

export interface ArtistEvolution {
  success: boolean;
  artist_id: string;
  artist_name: string;
  evolution: Array<{
    period: string;
    albums: Array<{
      name: string;
      release_date: string;
      popularity: number;
      genres: string[];
    }>;
    style_changes: string[];
    popularity_trend: 'increasing' | 'decreasing' | 'stable';
  }>;
  summary: string;
}

export interface TrendsResponse {
  success: boolean;
  trends: Array<{
    track_id: string;
    track_name: string;
    artists: string[];
    popularity: number;
    trend_direction: 'up' | 'down' | 'stable';
    change_percentage: number;
  }>;
  time_range: string;
  total: number;
}

