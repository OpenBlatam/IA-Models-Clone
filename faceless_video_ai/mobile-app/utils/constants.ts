/**
 * Application Constants
 * Centralized constants for the entire application
 */

// App Information
export const APP_NAME = 'Faceless Video AI';
export const APP_VERSION = '1.0.0';
export const APP_BUILD_NUMBER = '1';
export const APP_BUNDLE_ID = 'com.blatam.facelessvideoai';

// API Configuration
export const API_TIMEOUT = 30000; // 30 seconds
export const API_RETRY_ATTEMPTS = 3;
export const API_RETRY_DELAY = 1000; // 1 second

// Video Status Colors
export const VIDEO_STATUS_COLORS = {
  pending: '#FF9500',
  processing: '#007AFF',
  generating_images: '#007AFF',
  generating_audio: '#007AFF',
  adding_subtitles: '#007AFF',
  compositing: '#007AFF',
  completed: '#34C759',
  failed: '#FF3B30',
} as const;

// Video Status Labels
export const VIDEO_STATUS_LABELS = {
  pending: 'Pending',
  processing: 'Processing',
  generating_images: 'Generating Images',
  generating_audio: 'Generating Audio',
  adding_subtitles: 'Adding Subtitles',
  compositing: 'Compositing',
  completed: 'Completed',
  failed: 'Failed',
} as const;

// Polling Intervals (in milliseconds)
export const POLLING_INTERVALS = {
  video_status: 2000, // 2 seconds for processing videos
  analytics: 60000, // 1 minute
  quota: 30000, // 30 seconds
  batch_status: 5000, // 5 seconds
  scheduled_jobs: 30000, // 30 seconds
} as const;

// Default Values
export const DEFAULT_VIDEO_CONFIG = {
  resolution: '1920x1080',
  fps: 30,
  style: 'realistic' as const,
  transition_duration: 0.5,
  image_duration: 3.0,
  background_color: '#000000',
} as const;

export const DEFAULT_AUDIO_CONFIG = {
  voice: 'neutral' as const,
  speed: 1.0,
  pitch: 1.0,
  volume: 1.0,
  background_music: false,
  music_volume: 0.3,
} as const;

export const DEFAULT_SUBTITLE_CONFIG = {
  enabled: true,
  style: 'modern' as const,
  font_size: 48,
  font_color: '#FFFFFF',
  background_color: '#00000080',
  position: 'bottom' as const,
  animation: true,
  max_chars_per_line: 42,
  fade_in: true,
  fade_out: true,
} as const;

// Limits
export const LIMITS = {
  script_min_length: 10,
  script_max_length: 10000,
  batch_max_videos: 50,
  file_max_size_mb: 500,
  file_max_size_bytes: 500 * 1024 * 1024,
  password_min_length: 8,
  password_max_length: 128,
  email_max_length: 254,
  username_min_length: 3,
  username_max_length: 30,
  search_min_length: 1,
  search_max_length: 100,
  title_max_length: 200,
  description_max_length: 1000,
} as const;

// Storage Keys
export const STORAGE_KEYS = {
  auth_token: 'auth_token',
  api_key: 'api_key',
  theme: 'theme',
  language: 'language',
  user_preferences: 'user_preferences',
  cache_timestamp: 'cache_timestamp',
  onboarding_completed: 'onboarding_completed',
  last_sync: 'last_sync',
  user_settings: 'user_settings',
} as const;

// Video Styles
export const VIDEO_STYLES = {
  realistic: 'Realistic',
  animated: 'Animated',
  abstract: 'Abstract',
  minimalist: 'Minimalist',
  dynamic: 'Dynamic',
} as const;

// Audio Voices
export const AUDIO_VOICES = {
  male_1: 'Male Voice 1',
  male_2: 'Male Voice 2',
  female_1: 'Female Voice 1',
  female_2: 'Female Voice 2',
  neutral: 'Neutral Voice',
} as const;

// Subtitle Styles
export const SUBTITLE_STYLES = {
  simple: 'Simple',
  modern: 'Modern',
  bold: 'Bold',
  elegant: 'Elegant',
  minimal: 'Minimal',
} as const;

// Resolution Options
export const RESOLUTIONS = {
  '720p': { width: 1280, height: 720, label: '720p HD' },
  '1080p': { width: 1920, height: 1080, label: '1080p Full HD' },
  '1440p': { width: 2560, height: 1440, label: '1440p 2K' },
  '2160p': { width: 3840, height: 2160, label: '2160p 4K' },
} as const;

// FPS Options
export const FPS_OPTIONS = [24, 30, 60] as const;

// Output Formats
export const OUTPUT_FORMATS = {
  mp4: 'MP4',
  webm: 'WebM',
  mov: 'MOV',
} as const;

// Output Quality
export const OUTPUT_QUALITY = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  ultra: 'Ultra',
} as const;

// Languages
export const LANGUAGES = {
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  de: 'German',
  it: 'Italian',
  pt: 'Portuguese',
  ja: 'Japanese',
  ko: 'Korean',
  zh: 'Chinese',
  ru: 'Russian',
} as const;

// Timeouts
export const TIMEOUTS = {
  api_request: 30000, // 30 seconds
  file_upload: 300000, // 5 minutes
  video_generation: 3600000, // 1 hour
  image_generation: 300000, // 5 minutes
  audio_generation: 600000, // 10 minutes
  connection: 10000, // 10 seconds
} as const;

// Debounce/Throttle Delays
export const DELAYS = {
  search: 300, // 300ms
  input: 500, // 500ms
  scroll: 100, // 100ms
  resize: 250, // 250ms
  api_call: 1000, // 1 second
} as const;

// Cache Durations (in milliseconds)
export const CACHE_DURATIONS = {
  templates: 3600000, // 1 hour
  analytics: 300000, // 5 minutes
  quota: 60000, // 1 minute
  user_profile: 1800000, // 30 minutes
  video_list: 60000, // 1 minute
  search_results: 300000, // 5 minutes
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  network_error: 'Network error. Please check your connection.',
  timeout_error: 'Request timed out. Please try again.',
  server_error: 'Server error. Please try again later.',
  unauthorized: 'Unauthorized. Please log in again.',
  forbidden: 'Access forbidden.',
  not_found: 'Resource not found.',
  validation_error: 'Validation error. Please check your input.',
  unknown_error: 'An unexpected error occurred.',
  video_not_found: 'Video not found.',
  video_generation_failed: 'Video generation failed.',
  file_too_large: 'File is too large.',
  invalid_format: 'Invalid file format.',
  quota_exceeded: 'Quota exceeded.',
  rate_limit_exceeded: 'Rate limit exceeded. Please try again later.',
} as const;

// Success Messages
export const SUCCESS_MESSAGES = {
  video_generated: 'Video generated successfully!',
  video_deleted: 'Video deleted successfully.',
  video_shared: 'Video shared successfully.',
  settings_saved: 'Settings saved successfully.',
  profile_updated: 'Profile updated successfully.',
  template_created: 'Template created successfully.',
  feedback_submitted: 'Thank you for your feedback!',
} as const;

// Animation Durations
export const ANIMATION_DURATIONS = {
  fast: 150,
  normal: 300,
  slow: 500,
  very_slow: 1000,
} as const;

// Spacing
export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

// Border Radius
export const BORDER_RADIUS = {
  none: 0,
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
} as const;

// Font Sizes
export const FONT_SIZES = {
  xs: 12,
  sm: 14,
  md: 16,
  lg: 18,
  xl: 20,
  '2xl': 24,
  '3xl': 28,
  '4xl': 32,
  '5xl': 36,
} as const;

// Font Weights
export const FONT_WEIGHTS = {
  normal: '400' as const,
  medium: '500' as const,
  semibold: '600' as const,
  bold: '700' as const,
} as const;

// Z-Index Layers
export const Z_INDEX = {
  base: 0,
  dropdown: 1000,
  sticky: 1020,
  fixed: 1030,
  modal_backdrop: 1040,
  modal: 1050,
  popover: 1060,
  tooltip: 1070,
  toast: 1080,
  loading_overlay: 1090,
} as const;

// Breakpoints
export const BREAKPOINTS = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1400,
} as const;

// Screen Sizes
export const SCREEN_SIZES = {
  small_phone: 375,
  phone: 414,
  tablet: 768,
  large_tablet: 1024,
} as const;

// Platform Constants
export const PLATFORMS = {
  ios: 'ios',
  android: 'android',
  web: 'web',
} as const;

// File Types
export const FILE_TYPES = {
  image: ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
  video: ['mp4', 'webm', 'mov', 'avi', 'mkv'],
  audio: ['mp3', 'wav', 'aac', 'ogg', 'm4a'],
  document: ['pdf', 'doc', 'docx', 'txt', 'md'],
} as const;

// MIME Types
export const MIME_TYPES = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/gif': 'gif',
  'image/webp': 'webp',
  'video/mp4': 'mp4',
  'video/webm': 'webm',
  'video/quicktime': 'mov',
  'audio/mpeg': 'mp3',
  'audio/wav': 'wav',
  'application/pdf': 'pdf',
} as const;

// Regex Patterns
export const REGEX_PATTERNS = {
  email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  url: /^https?:\/\/.+/,
  phone: /^\+?[\d\s-()]+$/,
  password: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
  uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
  hex_color: /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/,
  resolution: /^\d+x\d+$/,
} as const;

// Validation Rules
export const VALIDATION_RULES = {
  email: {
    required: true,
    pattern: REGEX_PATTERNS.email,
    message: 'Please enter a valid email address',
  },
  password: {
    required: true,
    minLength: LIMITS.password_min_length,
    maxLength: LIMITS.password_max_length,
    pattern: REGEX_PATTERNS.password,
    message: 'Password must contain uppercase, lowercase, number and special character',
  },
  script: {
    required: true,
    minLength: LIMITS.script_min_length,
    maxLength: LIMITS.script_max_length,
    message: `Script must be between ${LIMITS.script_min_length} and ${LIMITS.script_max_length} characters`,
  },
} as const;

// HTTP Status Codes
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
} as const;

// HTTP Methods
export const HTTP_METHODS = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  PATCH: 'PATCH',
  DELETE: 'DELETE',
} as const;

// Content Types
export const CONTENT_TYPES = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  URL_ENCODED: 'application/x-www-form-urlencoded',
  TEXT: 'text/plain',
  HTML: 'text/html',
} as const;

// Feature Flags
export const FEATURE_FLAGS = {
  ENABLE_BATCH_GENERATION: true,
  ENABLE_TEMPLATES: true,
  ENABLE_ANALYTICS: true,
  ENABLE_SHARING: true,
  ENABLE_SCHEDULING: true,
  ENABLE_WATERMARKING: true,
  ENABLE_TRANSCRIPTION: true,
  ENABLE_VISUAL_EFFECTS: true,
} as const;

// Notification Types
export const NOTIFICATION_TYPES = {
  VIDEO_COMPLETED: 'video_completed',
  VIDEO_FAILED: 'video_failed',
  QUOTA_WARNING: 'quota_warning',
  SYSTEM_UPDATE: 'system_update',
} as const;

// Share Permissions
export const SHARE_PERMISSIONS = {
  VIEW: 'view',
  EDIT: 'edit',
  ADMIN: 'admin',
} as const;

// Schedule Repeat Options
export const SCHEDULE_REPEAT = {
  NONE: 'none',
  DAILY: 'daily',
  WEEKLY: 'weekly',
  MONTHLY: 'monthly',
} as const;

// Export Platforms
export const EXPORT_PLATFORMS = {
  YOUTUBE: 'youtube',
  INSTAGRAM: 'instagram',
  TIKTOK: 'tiktok',
  FACEBOOK: 'facebook',
  TWITTER: 'twitter',
  LINKEDIN: 'linkedin',
} as const;

// Music Styles
export const MUSIC_STYLES = {
  UPBEAT: 'upbeat',
  CALM: 'calm',
  EPIC: 'epic',
  CORPORATE: 'corporate',
  TECH: 'tech',
  NATURE: 'nature',
} as const;

// Alert Severity
export const ALERT_SEVERITY = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

// Backup Types
export const BACKUP_TYPES = {
  FULL: 'full',
  INCREMENTAL: 'incremental',
  METADATA_ONLY: 'metadata_only',
} as const;

// Export Formats
export const EXPORT_FORMATS = {
  CSV: 'csv',
  JSON: 'json',
  PDF: 'pdf',
  EXCEL: 'xlsx',
} as const;

// Date Formats
export const DATE_FORMATS = {
  SHORT: 'MMM dd, yyyy',
  LONG: 'MMMM dd, yyyy',
  TIME: 'HH:mm',
  DATETIME: 'MMM dd, yyyy HH:mm',
  ISO: 'yyyy-MM-ddTHH:mm:ss.SSSZ',
} as const;

// Number Formats
export const NUMBER_FORMATS = {
  DECIMAL: {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  },
  INTEGER: {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  },
  PERCENTAGE: {
    style: 'percent',
    minimumFractionDigits: 0,
    maximumFractionDigits: 1,
  },
  CURRENCY: {
    style: 'currency',
    currency: 'USD',
  },
} as const;

// Accessibility
export const ACCESSIBILITY = {
  MIN_TOUCH_TARGET: 44, // iOS minimum touch target
  MIN_TOUCH_TARGET_ANDROID: 48, // Android minimum touch target
  MIN_FONT_SIZE: 12,
  RECOMMENDED_FONT_SIZE: 16,
  MAX_CONTRAST_RATIO: 4.5, // WCAG AA
  ENHANCED_CONTRAST_RATIO: 7, // WCAG AAA
} as const;

// Performance Thresholds
export const PERFORMANCE_THRESHOLDS = {
  SLOW_RENDER_MS: 16, // One frame at 60fps
  SLOW_NETWORK_MS: 1000,
  LARGE_BUNDLE_SIZE_KB: 500,
  MAX_IMAGE_SIZE_MB: 10,
  MAX_VIDEO_SIZE_MB: 500,
} as const;

// Retry Configuration
export const RETRY_CONFIG = {
  MAX_RETRIES: 3,
  INITIAL_DELAY: 1000,
  MAX_DELAY: 10000,
  BACKOFF_MULTIPLIER: 2,
} as const;

// Pagination
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
  DEFAULT_PAGE: 1,
} as const;

// Search Configuration
export const SEARCH_CONFIG = {
  MIN_QUERY_LENGTH: 1,
  MAX_QUERY_LENGTH: 100,
  DEFAULT_LIMIT: 50,
  MAX_LIMIT: 200,
  SUGGESTIONS_LIMIT: 5,
} as const;

// Toast Configuration
export const TOAST_CONFIG = {
  DEFAULT_DURATION: 3000,
  SUCCESS_DURATION: 2000,
  ERROR_DURATION: 5000,
  WARNING_DURATION: 4000,
  INFO_DURATION: 3000,
} as const;

// Loading States
export const LOADING_STATES = {
  IDLE: 'idle',
  LOADING: 'loading',
  SUCCESS: 'success',
  ERROR: 'error',
} as const;

// Form States
export const FORM_STATES = {
  IDLE: 'idle',
  VALIDATING: 'validating',
  SUBMITTING: 'submitting',
  SUCCESS: 'success',
  ERROR: 'error',
} as const;

// Network States
export const NETWORK_STATES = {
  UNKNOWN: 'unknown',
  NONE: 'none',
  WIFI: 'wifi',
  CELLULAR: 'cellular',
  BLUETOOTH: 'bluetooth',
  ETHERNET: 'ethernet',
  WIMAX: 'wimax',
  VPN: 'vpn',
} as const;

// Theme Modes
export const THEME_MODES = {
  LIGHT: 'light',
  DARK: 'dark',
  AUTO: 'auto',
} as const;

// Storage Quotas
export const STORAGE_QUOTAS = {
  FREE_TIER_VIDEOS: 10,
  FREE_TIER_STORAGE_MB: 1000,
  PREMIUM_TIER_VIDEOS: 100,
  PREMIUM_TIER_STORAGE_MB: 10000,
  ENTERPRISE_TIER_VIDEOS: -1, // Unlimited
  ENTERPRISE_TIER_STORAGE_MB: -1, // Unlimited
} as const;

// Rate Limits
export const RATE_LIMITS = {
  FREE_TIER_PER_MINUTE: 10,
  FREE_TIER_PER_HOUR: 100,
  PREMIUM_TIER_PER_MINUTE: 60,
  PREMIUM_TIER_PER_HOUR: 1000,
  ENTERPRISE_TIER_PER_MINUTE: 300,
  ENTERPRISE_TIER_PER_HOUR: 10000,
} as const;

// Webhook Events
export const WEBHOOK_EVENTS = {
  VIDEO_COMPLETED: 'video.completed',
  VIDEO_FAILED: 'video.failed',
  VIDEO_PROCESSING: 'video.processing',
  BATCH_COMPLETED: 'batch.completed',
} as const;

// Analytics Events
export const ANALYTICS_EVENTS = {
  VIDEO_GENERATED: 'video_generated',
  VIDEO_DOWNLOADED: 'video_downloaded',
  VIDEO_SHARED: 'video_shared',
  TEMPLATE_USED: 'template_used',
  USER_REGISTERED: 'user_registered',
  USER_LOGGED_IN: 'user_logged_in',
  SEARCH_PERFORMED: 'search_performed',
  FEEDBACK_SUBMITTED: 'feedback_submitted',
} as const;

// Deep Link Paths
export const DEEP_LINK_PATHS = {
  VIDEO_DETAIL: '/video/:id',
  VIDEO_GENERATION: '/video/generate',
  TEMPLATE_DETAIL: '/template/:name',
  PROFILE: '/profile',
  SETTINGS: '/settings',
  ANALYTICS: '/analytics',
} as const;

// Keyboard Types
export const KEYBOARD_TYPES = {
  DEFAULT: 'default',
  NUMERIC: 'numeric',
  EMAIL: 'email-address',
  PHONE: 'phone-pad',
  DECIMAL: 'decimal-pad',
} as const;

// Return Key Types
export const RETURN_KEY_TYPES = {
  DONE: 'done',
  GO: 'go',
  NEXT: 'next',
  SEARCH: 'search',
  SEND: 'send',
} as const;

// Text Content Types (iOS)
export const TEXT_CONTENT_TYPES = {
  NONE: 'none',
  URL: 'URL',
  ADDRESS_CITY: 'addressCity',
  ADDRESS_CITY_STATE: 'addressCityAndState',
  ADDRESS_STATE: 'addressState',
  COUNTRY_NAME: 'countryName',
  CREDIT_CARD_NUMBER: 'creditCardNumber',
  EMAIL: 'emailAddress',
  FAMILY_NAME: 'familyName',
  FULL_STREET_ADDRESS: 'fullStreetAddress',
  GIVEN_NAME: 'givenName',
  JOB_TITLE: 'jobTitle',
  LOCATION: 'location',
  MIDDLE_NAME: 'middleName',
  NAME: 'name',
  NAME_PREFIX: 'namePrefix',
  NAME_SUFFIX: 'nameSuffix',
  NICKNAME: 'nickname',
  ORGANIZATION_NAME: 'organizationName',
  POSTAL_CODE: 'postalCode',
  TELEPHONE_NUMBER: 'telephoneNumber',
  USERNAME: 'username',
  PASSWORD: 'password',
  NEW_PASSWORD: 'newPassword',
  ONE_TIME_CODE: 'oneTimeCode',
} as const;

// Image Quality
export const IMAGE_QUALITY = {
  LOW: 0.3,
  MEDIUM: 0.6,
  HIGH: 0.8,
  VERY_HIGH: 1.0,
} as const;

// Video Quality Presets
export const VIDEO_QUALITY_PRESETS = {
  LOW: {
    bitrate: 1000000,
    resolution: '720p',
    fps: 24,
  },
  MEDIUM: {
    bitrate: 2500000,
    resolution: '1080p',
    fps: 30,
  },
  HIGH: {
    bitrate: 5000000,
    resolution: '1080p',
    fps: 60,
  },
  ULTRA: {
    bitrate: 10000000,
    resolution: '2160p',
    fps: 60,
  },
} as const;

// Animation Easing
export const ANIMATION_EASING = {
  LINEAR: 'linear',
  EASE_IN: 'ease-in',
  EASE_OUT: 'ease-out',
  EASE_IN_OUT: 'ease-in-out',
} as const;

// Gesture Handler Directions
export const GESTURE_DIRECTIONS = {
  UP: 'up',
  DOWN: 'down',
  LEFT: 'left',
  RIGHT: 'right',
} as const;

// Swipe Thresholds
export const SWIPE_THRESHOLDS = {
  VELOCITY: 0.5,
  DISTANCE: 50,
} as const;

// Haptic Feedback Types
export const HAPTIC_TYPES = {
  LIGHT: 'light',
  MEDIUM: 'medium',
  HEAVY: 'heavy',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error',
} as const;

// Error Codes
export const ERROR_CODES = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  AUTH_ERROR: 'AUTH_ERROR',
  PERMISSION_ERROR: 'PERMISSION_ERROR',
  QUOTA_ERROR: 'QUOTA_ERROR',
  RATE_LIMIT_ERROR: 'RATE_LIMIT_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
} as const;

// Type exports for better TypeScript support
export type VideoStatus = keyof typeof VIDEO_STATUS_COLORS;
export type VideoStyle = keyof typeof VIDEO_STYLES;
export type AudioVoice = keyof typeof AUDIO_VOICES;
export type SubtitleStyle = keyof typeof SUBTITLE_STYLES;
export type OutputFormat = keyof typeof OUTPUT_FORMATS;
export type OutputQuality = keyof typeof OUTPUT_QUALITY;
export type Language = keyof typeof LANGUAGES;
export type ThemeMode = keyof typeof THEME_MODES;
export type SharePermission = keyof typeof SHARE_PERMISSIONS;
export type ScheduleRepeat = keyof typeof SCHEDULE_REPEAT;
export type ExportPlatform = keyof typeof EXPORT_PLATFORMS;
export type MusicStyle = keyof typeof MUSIC_STYLES;
export type AlertSeverity = keyof typeof ALERT_SEVERITY;
export type LoadingState = keyof typeof LOADING_STATES;
export type FormState = keyof typeof FORM_STATES;
export type NetworkState = keyof typeof NETWORK_STATES;
