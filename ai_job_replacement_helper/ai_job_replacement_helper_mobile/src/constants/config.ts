import Constants from 'expo-constants';

export const API_BASE_URL =
  Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8030';

export const API_VERSION = 'v1';
export const API_PREFIX = `/api/${API_VERSION}`;

export const ENDPOINTS = {
  // Auth
  AUTH: {
    REGISTER: '/api/v1/auth/register',
    LOGIN: '/api/v1/auth/login',
    LOGOUT: '/api/v1/auth/logout',
    VERIFY: '/api/v1/auth/verify',
    USER: '/api/v1/auth/user',
  },
  // Gamification
  GAMIFICATION: {
    PROGRESS: '/api/v1/gamification/progress',
    POINTS: '/api/v1/gamification/points',
    LEADERBOARD: '/api/v1/gamification/leaderboard',
    BADGES: '/api/v1/gamification/badges',
  },
  // Steps
  STEPS: {
    ROADMAP: '/api/v1/steps/roadmap',
    PROGRESS: '/api/v1/steps/progress',
    START: '/api/v1/steps/start',
    COMPLETE: '/api/v1/steps/complete',
  },
  // Jobs
  JOBS: {
    SEARCH: '/api/v1/jobs/search',
    SWIPE: '/api/v1/jobs/swipe',
    APPLY: '/api/v1/jobs/apply',
    SAVED: '/api/v1/jobs/saved',
    LIKED: '/api/v1/jobs/liked',
    MATCHES: '/api/v1/jobs/matches',
    STATISTICS: '/api/v1/jobs/statistics',
  },
  // Recommendations
  RECOMMENDATIONS: {
    SKILLS: '/api/v1/recommendations/skills',
    JOBS: '/api/v1/recommendations/jobs',
    NEXT_STEPS: '/api/v1/recommendations/next-steps',
  },
  // Notifications
  NOTIFICATIONS: {
    LIST: '/api/v1/notifications',
    UNREAD_COUNT: '/api/v1/notifications/unread-count',
    MARK_READ: '/api/v1/notifications/mark-read',
    MARK_ALL_READ: '/api/v1/notifications/mark-all-read',
  },
  // Mentoring
  MENTORING: {
    START: '/api/v1/mentoring/start',
    ASK: '/api/v1/mentoring/ask',
    CAREER_ADVICE: '/api/v1/mentoring/career-advice',
    INTERVIEW_TIPS: '/api/v1/mentoring/interview-tips',
    MOTIVATION: '/api/v1/mentoring/motivation',
  },
  // CV Analyzer
  CV: {
    ANALYZE: '/api/v1/cv/analyze',
  },
  // Interview
  INTERVIEW: {
    START: '/api/v1/interview/start',
    ANSWER: '/api/v1/interview/answer',
    COMPLETE: '/api/v1/interview/complete',
  },
  // Challenges
  CHALLENGES: {
    AVAILABLE: '/api/v1/challenges/available',
    START: '/api/v1/challenges/start',
    PROGRESS: '/api/v1/challenges/progress',
    COMPLETE: '/api/v1/challenges/complete',
  },
  // Dashboard
  DASHBOARD: {
    MAIN: '/api/v1/dashboard',
    METRICS: '/api/v1/dashboard/metrics',
    ACTIVITY: '/api/v1/dashboard/activity',
  },
  // Content Generator
  CONTENT: {
    COVER_LETTER: '/api/v1/content/cover-letter',
    LINKEDIN_POST: '/api/v1/content/linkedin-post',
    FOLLOW_UP_EMAIL: '/api/v1/content/follow-up-email',
    IMPROVE_TEXT: '/api/v1/content/improve-text',
  },
  // Job Alerts
  JOB_ALERTS: {
    CREATE: '/api/v1/job-alerts/create',
    LIST: '/api/v1/job-alerts',
    CHECK: '/api/v1/job-alerts/check',
  },
  // Health
  HEALTH: {
    BASIC: '/health',
    DETAILED: '/health/detailed',
  },
} as const;

export const STORAGE_KEYS = {
  SESSION_ID: '@session_id',
  USER_ID: '@user_id',
  USER_DATA: '@user_data',
  THEME: '@theme',
} as const;


