export const AUTH_STORAGE_KEYS = {
  SESSION_ID: '@auth_session_id',
  USER_ID: '@auth_user_id',
  USER_DATA: '@auth_user_data',
} as const;

export const AUTH_ERRORS = {
  INVALID_CREDENTIALS: 'Invalid email or password',
  NETWORK_ERROR: 'Network error. Please check your connection.',
  UNKNOWN_ERROR: 'An unexpected error occurred',
  SESSION_EXPIRED: 'Your session has expired. Please login again.',
} as const;


