import axios from 'axios';

const MUSIC_API_URL = process.env.NEXT_PUBLIC_MUSIC_API_URL || 'http://localhost:8010';

const musicApi = axios.create({
  baseURL: `${MUSIC_API_URL}/music`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Track {
  id: string;
  name: string;
  artists: string[];
  album: string;
  duration_ms: number;
  preview_url?: string;
  popularity: number;
  external_urls?: {
    spotify?: string;
  };
  images?: Array<{ url: string; height?: number; width?: number }>;
}

export interface TrackSearchResponse {
  success: boolean;
  query: string;
  results: Track[];
  total: number;
}

export interface MusicalAnalysis {
  key_signature: string;
  root_note: string;
  mode: string;
  tempo: {
    bpm: number;
    category: string;
  };
  time_signature: string;
  scale: {
    name: string;
    notes: string[];
  };
}

export interface TechnicalAnalysis {
  energy: { value: number; description: string };
  danceability: { value: number; description: string };
  valence: { value: number; description: string };
  acousticness: { value: number; description: string };
  instrumentalness: { value: number; description: string };
  liveness: { value: number; description: string };
  loudness: { value: number; description: string };
}

export interface MusicAnalysisResponse {
  success: boolean;
  track_basic_info: {
    name: string;
    artists: string[];
    album: string;
    duration_seconds: number;
  };
  musical_analysis: MusicalAnalysis;
  technical_analysis: TechnicalAnalysis;
  composition_analysis?: any;
  educational_insights?: any;
  coaching?: any;
}

export interface ComparisonResponse {
  success: boolean;
  comparison: {
    key_signatures: {
      all_same: boolean;
      keys: string[];
      most_common: string;
    };
    tempos: {
      average: number;
      min: number;
      max: number;
      range: number;
    };
  };
  similarities: any[];
  differences: any[];
  recommendations: any[];
}

// API Functions
export const musicApiService = {
  // Search tracks
  searchTracks: async (query: string, limit: number = 10): Promise<TrackSearchResponse> => {
    const response = await musicApi.post<TrackSearchResponse>('/search', {
      query,
      limit,
    });
    return response.data;
  },

  // Analyze track
  analyzeTrack: async (
    trackId?: string,
    trackName?: string,
    includeCoaching: boolean = true
  ): Promise<MusicAnalysisResponse> => {
    const response = await musicApi.post<MusicAnalysisResponse>('/analyze', {
      ...(trackId && { track_id: trackId }),
      ...(trackName && { track_name: trackName }),
      include_coaching: includeCoaching,
    });
    return response.data;
  },

  // Get track info
  getTrackInfo: async (trackId: string): Promise<any> => {
    const response = await musicApi.get(`/track/${trackId}/info`);
    return response.data;
  },

  // Get audio features
  getAudioFeatures: async (trackId: string): Promise<any> => {
    const response = await musicApi.get(`/track/${trackId}/audio-features`);
    return response.data;
  },

  // Get audio analysis
  getAudioAnalysis: async (trackId: string): Promise<any> => {
    const response = await musicApi.get(`/track/${trackId}/audio-analysis`);
    return response.data;
  },

  // Compare tracks
  compareTracks: async (trackIds: string[]): Promise<ComparisonResponse> => {
    const response = await musicApi.post<ComparisonResponse>('/compare', {
      track_ids: trackIds,
    });
    return response.data;
  },

  // Get recommendations
  getRecommendations: async (trackId: string, limit: number = 20): Promise<any> => {
    const response = await musicApi.get(`/track/${trackId}/recommendations`, {
      params: { limit },
    });
    return response.data;
  },

  // Get coaching
  getCoaching: async (trackName: string): Promise<any> => {
    const response = await musicApi.post('/coaching', {
      track_name: trackName,
    });
    return response.data;
  },

  // ML Analysis
  analyzeComprehensive: async (trackId: string): Promise<any> => {
    const response = await musicApi.post(`/ml/analyze-comprehensive`, null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  // Health check
  healthCheck: async (): Promise<any> => {
    const response = await musicApi.get('/health');
    return response.data;
  },

  // Get history
  getHistory: async (userId?: string, limit: number = 50): Promise<any> => {
    const response = await musicApi.get('/history', {
      params: { user_id: userId, limit },
    });
    return response.data;
  },

  // Get favorites
  getFavorites: async (userId?: string): Promise<any> => {
    const response = await musicApi.get('/favorites', {
      params: { user_id: userId },
    });
    return response.data;
  },

  // Add to favorites
  addToFavorites: async (
    userId: string,
    trackId: string,
    trackName: string,
    artists: string[]
  ): Promise<any> => {
    const response = await musicApi.post('/favorites', null, {
      params: {
        user_id: userId,
        track_id: trackId,
        track_name: trackName,
        artists: artists.join(','),
      },
    });
    return response.data;
  },

  // Get analytics
  getAnalytics: async (): Promise<any> => {
    const response = await musicApi.get('/analytics');
    return response.data;
  },

  // ML Predictions
  predictGenre: async (trackId: string): Promise<any> => {
    const response = await musicApi.post(`/ml/predict/genre`, null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  predictMultiTask: async (trackId: string): Promise<any> => {
    const response = await musicApi.post(`/ml/predict/multi-task`, null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  compareTracksML: async (trackIds: string[], comparisonType: string = 'all'): Promise<any> => {
    const response = await musicApi.post('/ml/compare-tracks', {
      track_ids: trackIds,
    }, {
      params: { comparison_type: comparisonType },
    });
    return response.data;
  },

  // Contextual Recommendations
  getContextualRecommendations: async (
    trackId: string,
    context?: string,
    limit: number = 20
  ): Promise<any> => {
    const response = await musicApi.post('/recommendations/contextual', {
      track_id: trackId,
      context,
      limit,
    });
    return response.data;
  },

  getRecommendationsByMood: async (mood: string, limit: number = 20): Promise<any> => {
    const response = await musicApi.post('/recommendations/mood', {
      mood,
      limit,
    });
    return response.data;
  },

  getRecommendationsByActivity: async (activity: string, limit: number = 20): Promise<any> => {
    const response = await musicApi.post('/recommendations/activity', {
      activity,
      limit,
    });
    return response.data;
  },

  getRecommendationsByTimeOfDay: async (timeOfDay: string, limit: number = 20): Promise<any> => {
    const response = await musicApi.post('/recommendations/time-of-day', {
      time_of_day: timeOfDay,
      limit,
    });
    return response.data;
  },

  // Trends
  getTrends: async (): Promise<any> => {
    const response = await musicApi.get('/trends/popularity');
    return response.data;
  },

  predictSuccess: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/predict/success', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  // Discovery
  getSimilarArtists: async (artistName: string, limit: number = 10): Promise<any> => {
    const response = await musicApi.post('/discovery/similar-artists', {
      artist_name: artistName,
      limit,
    });
    return response.data;
  },

  getUndergroundTracks: async (limit: number = 20): Promise<any> => {
    const response = await musicApi.post('/discovery/underground', {
      limit,
    });
    return response.data;
  },

  // Playlists
  createPlaylist: async (userId: string, name: string, isPublic: boolean = false): Promise<any> => {
    const response = await musicApi.post('/playlists', null, {
      params: {
        user_id: userId,
        name,
        is_public: isPublic,
      },
    });
    return response.data;
  },

  getPlaylists: async (userId?: string, publicOnly: boolean = false): Promise<any> => {
    const response = await musicApi.get('/playlists', {
      params: {
        user_id: userId,
        public_only: publicOnly,
      },
    });
    return response.data;
  },

  analyzePlaylist: async (trackIds: string[]): Promise<any> => {
    const response = await musicApi.post('/playlists/analyze', {
      track_ids: trackIds,
    });
    return response.data;
  },

  // Temporal Analysis
  getTemporalStructure: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/temporal/structure', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  getTemporalEnergy: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/temporal/energy', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  // Quality Analysis
  analyzeQuality: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/quality/analyze', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  // Export
  exportAnalysis: async (
    trackId: string,
    format: 'json' | 'text' | 'markdown' = 'json',
    includeCoaching: boolean = true
  ): Promise<any> => {
    const response = await musicApi.post(`/export/${trackId}`, null, {
      params: {
        format,
        include_coaching: includeCoaching,
      },
    });
    return response.data;
  },

  // Remove from favorites
  removeFromFavorites: async (userId: string, trackId: string): Promise<any> => {
    const response = await musicApi.delete(`/favorites/${trackId}`, {
      params: { user_id: userId },
    });
    return response.data;
  },

  // Collaboration Network
  getCollaborationNetwork: async (artistNames: string[]): Promise<any> => {
    const response = await musicApi.post('/collaborations/network', {
      artist_names: artistNames,
    });
    return response.data;
  },

  // Analyze Collaborations
  analyzeCollaborations: async (artistName: string): Promise<any> => {
    const response = await musicApi.post('/collaborations/analyze', null, {
      params: { artist_name: artistName },
    });
    return response.data;
  },

  // Covers and Remixes
  findCoversRemixes: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/covers/find', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  analyzeCover: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/covers/analyze', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  analyzeRemix: async (trackId: string): Promise<any> => {
    const response = await musicApi.post('/remixes/analyze', null, {
      params: { track_id: trackId },
    });
    return response.data;
  },

  // Artist Analysis
  getArtistEvolution: async (artistName: string): Promise<any> => {
    const response = await musicApi.post('/artists/evolution', null, {
      params: { artist_name: artistName },
    });
    return response.data;
  },

  compareArtists: async (artistNames: string[]): Promise<any> => {
    const response = await musicApi.post('/artists/compare', {
      artist_names: artistNames,
    });
    return response.data;
  },

  // Notifications
  getNotifications: async (userId: string, unreadOnly: boolean = false, limit: number = 50): Promise<any> => {
    const response = await musicApi.get('/notifications', {
      params: {
        user_id: userId,
        unread_only: unreadOnly,
        limit,
      },
    });
    return response.data;
  },

  // Tags
  getTags: async (resourceId: string, resourceType: string): Promise<any> => {
    const response = await musicApi.get(`/tags/${resourceId}`, {
      params: { resource_type: resourceType },
    });
    return response.data;
  },

  addTag: async (resourceId: string, resourceType: string, tags: string[]): Promise<any> => {
    const response = await musicApi.post('/tags', null, {
      params: {
        resource_id: resourceId,
        resource_type: resourceType,
        tags: tags.join(','),
      },
    });
    return response.data;
  },

  removeTag: async (resourceId: string, resourceType: string, tags: string[]): Promise<any> => {
    const response = await musicApi.delete('/tags', {
      params: {
        resource_id: resourceId,
        resource_type: resourceType,
        tags: tags.join(','),
      },
    });
    return response.data;
  },

  // Lyrics
  getLyrics: async (trackId: string): Promise<any> => {
    const response = await musicApi.get(`/lyrics/${trackId}`);
    return response.data;
  },

  // Notes
  getNotes: async (resourceId: string, resourceType: string): Promise<any> => {
    const response = await musicApi.get(`/notes/${resourceId}`, {
      params: { resource_type: resourceType },
    });
    return response.data;
  },

  addNote: async (resourceId: string, resourceType: string, content: string): Promise<any> => {
    const response = await musicApi.post('/notes', null, {
      params: {
        resource_id: resourceId,
        resource_type: resourceType,
        content,
      },
    });
    return response.data;
  },

  updateNote: async (noteId: string, content: string): Promise<any> => {
    const response = await musicApi.put(`/notes/${noteId}`, null, {
      params: { content },
    });
    return response.data;
  },

  deleteNote: async (noteId: string): Promise<any> => {
    const response = await musicApi.delete(`/notes/${noteId}`);
    return response.data;
  },

  // Bookmarks
  getBookmarks: async (userId: string): Promise<any> => {
    const response = await musicApi.get('/bookmarks', {
      params: { user_id: userId },
    });
    return response.data;
  },

  addBookmark: async (userId: string, trackId: string, folderId?: string): Promise<any> => {
    const response = await musicApi.post('/bookmarks', null, {
      params: {
        user_id: userId,
        track_id: trackId,
        folder_id: folderId,
      },
    });
    return response.data;
  },

  removeBookmark: async (bookmarkId: string): Promise<any> => {
    const response = await musicApi.delete(`/bookmarks/${bookmarkId}`);
    return response.data;
  },

  getBookmarkFolders: async (userId: string): Promise<any> => {
    const response = await musicApi.get('/bookmarks/folders', {
      params: { user_id: userId },
    });
    return response.data;
  },

  // Rating
  rateTrack: async (trackId: string, rating: number): Promise<any> => {
    const response = await musicApi.post('/rating', null, {
      params: {
        track_id: trackId,
        rating,
      },
    });
    return response.data;
  },

  getTrackRating: async (trackId: string): Promise<any> => {
    const response = await musicApi.get(`/rating/${trackId}`);
    return response.data;
  },

  // Comments
  getComments: async (resourceId: string, resourceType: string): Promise<any> => {
    const response = await musicApi.get(`/comments/${resourceId}`, {
      params: { resource_type: resourceType },
    });
    return response.data;
  },

  addComment: async (resourceId: string, resourceType: string, content: string): Promise<any> => {
    const response = await musicApi.post('/comments', null, {
      params: {
        resource_id: resourceId,
        resource_type: resourceType,
        content,
      },
    });
    return response.data;
  },

  updateComment: async (commentId: string, content: string): Promise<any> => {
    const response = await musicApi.put(`/comments/${commentId}`, null, {
      params: { content },
    });
    return response.data;
  },

  deleteComment: async (commentId: string): Promise<any> => {
    const response = await musicApi.delete(`/comments/${commentId}`);
    return response.data;
  },
};

