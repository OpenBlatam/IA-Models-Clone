export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    CAMERA: {
      INITIALIZE: '/api/quality-control/camera/initialize',
      INFO: '/api/quality-control/camera/info',
      SETTINGS: '/api/quality-control/camera/settings',
      CAPTURE: '/api/quality-control/camera/capture',
    },
    INSPECTION: {
      START: '/api/quality-control/inspection/start',
      STOP: '/api/quality-control/inspection/stop',
      INSPECT: '/api/quality-control/inspection/inspect',
      BATCH: '/api/quality-control/inspection/batch',
    },
    ALERTS: {
      RECENT: '/api/quality-control/alerts/recent',
      STATISTICS: '/api/quality-control/alerts/statistics',
    },
    DETECTION: {
      SETTINGS: '/api/quality-control/detection/settings',
    },
    REPORTS: {
      GENERATE: '/api/quality-control/reports/generate',
    },
    VIDEO: {
      ANALYZE: '/api/quality-control/video/analyze',
    },
  },
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 1,
} as const;

