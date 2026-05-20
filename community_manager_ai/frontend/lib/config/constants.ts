/**
 * Application constants
 * Centralizes all constant values used throughout the application
 */

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  POSTS: '/posts',
  MEMES: '/memes',
  CALENDAR: '/calendar',
  PLATFORMS: '/platforms',
  ANALYTICS: '/analytics',
  DASHBOARD: '/dashboard',
  TEMPLATES: '/templates',
} as const;

/**
 * Query keys for React Query
 */
export const QUERY_KEYS = {
  posts: {
    all: ['posts'] as const,
    lists: () => [...QUERY_KEYS.posts.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...QUERY_KEYS.posts.lists(), filters] as const,
    details: () => [...QUERY_KEYS.posts.all, 'detail'] as const,
    detail: (id: string) => [...QUERY_KEYS.posts.details(), id] as const,
  },
  memes: {
    all: ['memes'] as const,
    lists: () => [...QUERY_KEYS.memes.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...QUERY_KEYS.memes.lists(), filters] as const,
    details: () => [...QUERY_KEYS.memes.all, 'detail'] as const,
    detail: (id: string) => [...QUERY_KEYS.memes.details(), id] as const,
  },
  calendar: {
    all: ['calendar'] as const,
    events: (filters: Record<string, unknown>) => [...QUERY_KEYS.calendar.all, 'events', filters] as const,
    daily: (date: string) => [...QUERY_KEYS.calendar.all, 'daily', date] as const,
    weekly: (startDate?: string) => [...QUERY_KEYS.calendar.all, 'weekly', startDate] as const,
  },
  platforms: {
    all: ['platforms'] as const,
    list: () => [...QUERY_KEYS.platforms.all, 'list'] as const,
    detail: (platform: string) => [...QUERY_KEYS.platforms.all, 'detail', platform] as const,
  },
  analytics: {
    all: ['analytics'] as const,
    platform: (platform: string, days: number) => [...QUERY_KEYS.analytics.all, 'platform', platform, days] as const,
    post: (postId: string, platform?: string) => [...QUERY_KEYS.analytics.all, 'post', postId, platform] as const,
    bestPerforming: (platform?: string, limit?: number) => [...QUERY_KEYS.analytics.all, 'best-performing', platform, limit] as const,
    trends: (platform: string, days: number) => [...QUERY_KEYS.analytics.all, 'trends', platform, days] as const,
  },
  dashboard: {
    all: ['dashboard'] as const,
    overview: (days: number) => [...QUERY_KEYS.dashboard.all, 'overview', days] as const,
    engagement: (days: number) => [...QUERY_KEYS.dashboard.all, 'engagement', days] as const,
    upcomingPosts: (limit: number) => [...QUERY_KEYS.dashboard.all, 'upcoming-posts', limit] as const,
    recentActivity: (limit: number) => [...QUERY_KEYS.dashboard.all, 'recent-activity', limit] as const,
  },
  templates: {
    all: ['templates'] as const,
    lists: () => [...QUERY_KEYS.templates.all, 'list'] as const,
    list: () => [...QUERY_KEYS.templates.lists()] as const,
    details: () => [...QUERY_KEYS.templates.all, 'detail'] as const,
    detail: (id: string) => [...QUERY_KEYS.templates.details(), id] as const,
  },
} as const;

/**
 * Platform names
 */
export const PLATFORMS = {
  FACEBOOK: 'facebook',
  INSTAGRAM: 'instagram',
  TWITTER: 'twitter',
  LINKEDIN: 'linkedin',
  TIKTOK: 'tiktok',
  YOUTUBE: 'youtube',
} as const;

/**
 * Post statuses
 */
export const POST_STATUS = {
  SCHEDULED: 'scheduled',
  PUBLISHED: 'published',
  CANCELLED: 'cancelled',
} as const;

/**
 * Default values
 */
export const DEFAULTS = {
  QUERY_STALE_TIME: 5 * 60 * 1000, // 5 minutes
  QUERY_RETRY: 1,
  ANALYTICS_DAYS: 7,
  DASHBOARD_DAYS: 7,
  UPCOMING_POSTS_LIMIT: 10,
  RECENT_ACTIVITY_LIMIT: 20,
  BEST_PERFORMING_LIMIT: 10,
} as const;


