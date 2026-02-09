import Constants from 'expo-constants';

export const API_BASE_URL =
  Constants.expoConfig?.extra?.apiBaseUrl ||
  process.env.EXPO_PUBLIC_API_BASE_URL ||
  'http://localhost:8000';

export const API_VERSION = '/api/v1';

export const API_ENDPOINTS = {
  // Auth
  AUTH: {
    REGISTER: '/auth/register',
    LOGIN: '/auth/login',
    REFRESH: '/auth/refresh',
    ME: '/auth/me',
    API_KEY: '/auth/api-key',
    REGENERATE_API_KEY: '/auth/api-key/regenerate',
  },
  // Video Generation
  VIDEO: {
    GENERATE: '/generate',
    STATUS: (videoId: string) => `/status/${videoId}`,
    DOWNLOAD: (videoId: string) => `/videos/${videoId}/download`,
    DELETE: (videoId: string) => `/videos/${videoId}`,
    UPLOAD_SCRIPT: '/upload-script',
    WEBHOOK: (videoId: string) => `/videos/${videoId}/webhook`,
    VERSIONS: (videoId: string) => `/videos/${videoId}/versions`,
    VERSION: (videoId: string, version: number) =>
      `/videos/${videoId}/versions/${version}`,
    COMPARE_VERSIONS: (videoId: string) => `/videos/${videoId}/versions/compare`,
    EXPORT: (videoId: string) => `/videos/${videoId}/export`,
    WATERMARK: (videoId: string) => `/videos/${videoId}/watermark`,
    TRANSCRIBE: (videoId: string) => `/videos/${videoId}/transcribe`,
    KEN_BURNS: (videoId: string) => `/videos/${videoId}/effects/ken-burns`,
    SHARE: (videoId: string) => `/videos/${videoId}/share`,
    SHARES: (videoId: string) => `/videos/${videoId}/shares`,
    FEEDBACK: (videoId: string) => `/videos/${videoId}/feedback`,
    SCHEDULE: (videoId: string) => `/videos/${videoId}/schedule`,
  },
  // Batch
  BATCH: {
    GENERATE: '/batch/generate',
    STATUS: '/batch/status',
  },
  // Templates
  TEMPLATES: {
    LIST: '/templates',
    GET: (name: string) => `/templates/${name}`,
    GENERATE: (name: string) => `/templates/${name}/generate`,
  },
  // Custom Templates
  CUSTOM_TEMPLATES: {
    CREATE: '/custom-templates',
    LIST: '/custom-templates',
  },
  // Analytics
  ANALYTICS: '/analytics',
  // Recommendations
  RECOMMENDATIONS: '/recommendations',
  // Music
  MUSIC: {
    TRACKS: '/music/tracks',
  },
  // Platforms
  PLATFORMS: '/platforms',
  // Search
  SEARCH: {
    VIDEOS: '/search',
    SUGGESTIONS: '/search/suggestions',
  },
  // Shared Videos
  SHARED: {
    VIDEOS: '/shared-videos',
    BY_TOKEN: (token: string) => `/shared/${token}`,
  },
  // Scheduled
  SCHEDULED: {
    LIST: '/scheduled',
    CANCEL: (jobId: string) => `/scheduled/${jobId}`,
  },
  // Quota
  QUOTA: '/quota',
  // Admin
  ADMIN: {
    DASHBOARD: '/admin/dashboard',
    HEALTH: '/admin/health',
    METRICS: '/admin/metrics',
    PROFILES: '/admin/profiles',
    BACKUP: '/admin/backup',
    BACKUPS: '/admin/backups',
    RESTORE: (backupId: string) => `/admin/backups/${backupId}/restore`,
    CONFIG: '/admin/config',
  },
  // Alerts
  ALERTS: {
    LIST: '/alerts',
    ACKNOWLEDGE: (alertId: string) => `/alerts/${alertId}/acknowledge`,
    RESOLVE: (alertId: string) => `/alerts/${alertId}/resolve`,
  },
  // Export
  EXPORT: {
    VIDEOS: '/export/videos',
  },
  // Metrics
  METRICS: '/metrics/prometheus',
  // Health
  HEALTH: '/health',
  HEALTH_LIVE: '/health/live',
  HEALTH_READY: '/health/ready',
} as const;


