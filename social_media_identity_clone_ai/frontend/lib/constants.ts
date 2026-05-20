export const API_ENDPOINTS = {
  EXTRACT_PROFILE: '/extract-profile',
  BUILD_IDENTITY: '/build-identity',
  GENERATE_CONTENT: '/generate-content',
  GET_IDENTITY: (id: string) => `/identity/${id}`,
  GET_GENERATED_CONTENT: (id: string) => `/identity/${id}/generated-content`,
  TASKS: '/tasks',
  METRICS: '/metrics',
  DASHBOARD: '/dashboard',
  ALERTS: '/alerts',
  TEMPLATES: '/templates',
} as const;

export const PLATFORM_OPTIONS = [
  { value: 'tiktok', label: 'TikTok' },
  { value: 'instagram', label: 'Instagram' },
  { value: 'youtube', label: 'YouTube' },
] as const;

export const CONTENT_TYPE_OPTIONS = [
  { value: 'post', label: 'Post' },
  { value: 'video', label: 'Video' },
  { value: 'reel', label: 'Reel' },
  { value: 'story', label: 'Story' },
] as const;

export const REFETCH_INTERVALS = {
  TASKS: 5000,
  METRICS: 30000,
  DASHBOARD: 60000,
} as const;

export const DEBOUNCE_DELAYS = {
  SEARCH: 300,
  INPUT: 500,
} as const;

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 10,
  MAX_PAGE_SIZE: 100,
} as const;

export const VALIDATION = {
  MIN_USERNAME_LENGTH: 1,
  MAX_USERNAME_LENGTH: 255,
  MIN_PASSWORD_LENGTH: 8,
} as const;

export * from './constants/forms';

