export const APP_CONFIG = {
  NAME: 'Quality Control AI',
  VERSION: '1.0.0',
  DESCRIPTION: 'AI-powered quality control and defect detection system',
  QUERY_CONFIG: {
    STALE_TIME: 60 * 1000,
    REFETCH_ON_WINDOW_FOCUS: false,
    REFETCH_ON_MOUNT: false,
    RETRY: 1,
  },
  TOAST_CONFIG: {
    POSITION: 'top-right' as const,
    DURATION: 5000,
  },
} as const;

