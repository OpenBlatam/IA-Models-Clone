import Constants from 'expo-constants';

const API_BASE_URL = __DEV__
  ? Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8020'
  : Constants.expoConfig?.extra?.apiUrl || 'https://api.example.com';

export const API_CONFIG = {
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
};

export const API_ENDPOINTS = {
  // Generation
  GENERATE: '/api/v1/generate',
  GENERATE_BATCH: '/api/v1/generate/batch',
  GENERATION_TASK: (taskId: string) => `/api/v1/generate/task/${taskId}`,
  
  // Projects
  PROJECTS: '/api/v1/projects',
  PROJECT: (id: string) => `/api/v1/projects/${id}`,
  PROJECT_QUEUE_STATUS: '/api/v1/projects/queue/status',
  
  // Status & Monitoring
  STATUS: '/api/v1/status',
  STATS: '/api/v1/stats',
  QUEUE: '/api/v1/queue',
  
  // Control
  START: '/api/v1/start',
  STOP: '/api/v1/stop',
  
  // Export
  EXPORT_ZIP: '/api/v1/export/zip',
  EXPORT_TAR: '/api/v1/export/tar',
  
  // Validation
  VALIDATE: '/api/v1/validate',
  
  // Deployment
  DEPLOY_GENERATE: '/api/v1/deploy/generate',
  
  // GitHub
  GITHUB_CREATE: '/api/v1/github/create',
  GITHUB_PUSH: '/api/v1/github/push',
  
  // Clone
  CLONE: '/api/v1/clone',
  
  // Templates
  TEMPLATES: '/api/v1/templates',
  TEMPLATES_LIST: '/api/v1/templates/list',
  TEMPLATE: (name: string) => `/api/v1/templates/${name}`,
  
  // Search
  SEARCH: '/api/v1/search',
  SEARCH_STATS: '/api/v1/search/stats',
  
  // Webhooks
  WEBHOOKS: '/api/v1/webhooks',
  WEBHOOK: (id: string) => `/api/v1/webhooks/${id}`,
  
  // Cache
  CACHE_CLEAR: '/api/v1/cache/clear',
  CACHE_STATS: '/api/v1/cache/stats',
  
  // Rate Limit
  RATE_LIMIT: '/api/v1/rate-limit',
  
  // Auth
  AUTH_REGISTER: '/api/v1/auth/register',
  AUTH_LOGIN: '/api/v1/auth/login',
  AUTH_API_KEY: '/api/v1/auth/api-key',
  
  // Metrics
  METRICS: '/api/v1/metrics',
  METRICS_PROMETHEUS: '/api/v1/metrics/prometheus',
  
  // Backup
  BACKUP_CREATE: '/api/v1/backup/create',
  BACKUP_LIST: '/api/v1/backup/list',
  BACKUP_RESTORE: '/api/v1/backup/restore',
  BACKUP: (name: string) => `/api/v1/backup/${name}`,
  
  // Health
  HEALTH: '/health',
  HEALTH_DETAILED: '/health/detailed',
  
  // Version
  VERSION: '/api/version',
  
  // Analytics
  ANALYTICS_TRENDS: '/api/v1/analytics/trends',
  ANALYTICS_TOP_AI_TYPES: '/api/v1/analytics/top-ai-types',
  
  // Performance
  PERFORMANCE_STATS: '/api/v1/performance/stats',
  PERFORMANCE_OPTIMIZE: '/api/v1/performance/optimize',
};

