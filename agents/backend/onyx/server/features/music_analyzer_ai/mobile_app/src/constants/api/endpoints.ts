export const API_ENDPOINTS = {
  SEARCH: '/music/search',
  ANALYZE: '/music/analyze',
  ANALYZE_BY_ID: (trackId: string) => `/music/analyze/${trackId}`,
  TRACK_INFO: (trackId: string) => `/music/track/${trackId}/info`,
  AUDIO_FEATURES: (trackId: string) => `/music/track/${trackId}/audio-features`,
  AUDIO_ANALYSIS: (trackId: string) => `/music/track/${trackId}/audio-analysis`,
  RECOMMENDATIONS: (trackId: string) =>
    `/music/track/${trackId}/recommendations`,
  COMPARE: '/music/compare',
  COACHING: '/music/coaching',
  HEALTH: '/music/health',
  HISTORY: '/music/history',
  HISTORY_STATS: '/music/history/stats',
  FAVORITES: '/music/favorites',
  FAVORITES_STATS: '/music/favorites/stats',
  EXPORT: (trackId: string, format?: string) =>
    `/music/export/${trackId}${format ? `?format=${format}` : ''}`,
  CONTEXTUAL_RECOMMENDATIONS: '/music/recommendations/contextual',
  TIME_OF_DAY_RECOMMENDATIONS: '/music/recommendations/time-of-day',
  ACTIVITY_RECOMMENDATIONS: '/music/recommendations/activity',
  MOOD_RECOMMENDATIONS: '/music/recommendations/mood',
  DISCOVERY_SIMILAR_ARTISTS: '/music/discovery/similar-artists',
  DISCOVERY_UNDERGROUND: '/music/discovery/underground',
  DISCOVERY_MOOD_TRANSITION: '/music/discovery/mood-transition',
  DISCOVERY_FRESH: '/music/discovery/fresh',
  ARTIST_COMPARE: '/music/artists/compare',
  ARTIST_EVOLUTION: '/music/artists/evolution',
  TRENDS_POPULARITY: '/music/trends/popularity',
  TRENDS_ARTISTS: '/music/trends/artists',
} as const;


