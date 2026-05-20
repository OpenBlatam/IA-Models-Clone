# Constants Reference

Documentación completa de todas las constantes de la aplicación.

## 📋 Categorías

### 1. App Information
```tsx
APP_NAME: 'Faceless Video AI'
APP_VERSION: '1.0.0'
APP_BUILD_NUMBER: '1'
APP_BUNDLE_ID: 'com.blatam.facelessvideoai'
```

### 2. API Configuration
```tsx
API_TIMEOUT: 30000 // 30 seconds
API_RETRY_ATTEMPTS: 3
API_RETRY_DELAY: 1000 // 1 second
```

### 3. Video Status
```tsx
VIDEO_STATUS_COLORS // Colores para cada estado
VIDEO_STATUS_LABELS // Etiquetas legibles
```

### 4. Polling Intervals
```tsx
POLLING_INTERVALS = {
  video_status: 2000,      // 2 segundos
  analytics: 60000,         // 1 minuto
  quota: 30000,            // 30 segundos
  batch_status: 5000,      // 5 segundos
  scheduled_jobs: 30000,   // 30 segundos
}
```

### 5. Default Configurations
```tsx
DEFAULT_VIDEO_CONFIG    // Configuración por defecto de video
DEFAULT_AUDIO_CONFIG    // Configuración por defecto de audio
DEFAULT_SUBTITLE_CONFIG // Configuración por defecto de subtítulos
```

### 6. Limits
```tsx
LIMITS = {
  script_min_length: 10,
  script_max_length: 10000,
  batch_max_videos: 50,
  file_max_size_mb: 500,
  password_min_length: 8,
  // ... más límites
}
```

### 7. Storage Keys
```tsx
STORAGE_KEYS = {
  auth_token: 'auth_token',
  api_key: 'api_key',
  theme: 'theme',
  language: 'language',
  // ... más keys
}
```

### 8. Video Styles
```tsx
VIDEO_STYLES = {
  realistic: 'Realistic',
  animated: 'Animated',
  abstract: 'Abstract',
  minimalist: 'Minimalist',
  dynamic: 'Dynamic',
}
```

### 9. Audio Voices
```tsx
AUDIO_VOICES = {
  male_1: 'Male Voice 1',
  male_2: 'Male Voice 2',
  female_1: 'Female Voice 1',
  female_2: 'Female Voice 2',
  neutral: 'Neutral Voice',
}
```

### 10. Subtitle Styles
```tsx
SUBTITLE_STYLES = {
  simple: 'Simple',
  modern: 'Modern',
  bold: 'Bold',
  elegant: 'Elegant',
  minimal: 'Minimal',
}
```

### 11. Resolutions
```tsx
RESOLUTIONS = {
  '720p': { width: 1280, height: 720, label: '720p HD' },
  '1080p': { width: 1920, height: 1080, label: '1080p Full HD' },
  '1440p': { width: 2560, height: 1440, label: '1440p 2K' },
  '2160p': { width: 3840, height: 2160, label: '2160p 4K' },
}
```

### 12. FPS Options
```tsx
FPS_OPTIONS = [24, 30, 60]
```

### 13. Output Formats & Quality
```tsx
OUTPUT_FORMATS = { mp4: 'MP4', webm: 'WebM', mov: 'MOV' }
OUTPUT_QUALITY = { low: 'Low', medium: 'Medium', high: 'High', ultra: 'Ultra' }
```

### 14. Languages
```tsx
LANGUAGES = {
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  // ... más idiomas
}
```

### 15. Timeouts
```tsx
TIMEOUTS = {
  api_request: 30000,        // 30 segundos
  file_upload: 300000,        // 5 minutos
  video_generation: 3600000,  // 1 hora
  // ... más timeouts
}
```

### 16. Delays (Debounce/Throttle)
```tsx
DELAYS = {
  search: 300,      // 300ms
  input: 500,       // 500ms
  scroll: 100,      // 100ms
  resize: 250,      // 250ms
  api_call: 1000,   // 1 segundo
}
```

### 17. Cache Durations
```tsx
CACHE_DURATIONS = {
  templates: 3600000,      // 1 hora
  analytics: 300000,       // 5 minutos
  quota: 60000,           // 1 minuto
  user_profile: 1800000,  // 30 minutos
  // ... más duraciones
}
```

### 18. Error Messages
```tsx
ERROR_MESSAGES = {
  network_error: 'Network error. Please check your connection.',
  timeout_error: 'Request timed out. Please try again.',
  // ... más mensajes
}
```

### 19. Success Messages
```tsx
SUCCESS_MESSAGES = {
  video_generated: 'Video generated successfully!',
  video_deleted: 'Video deleted successfully.',
  // ... más mensajes
}
```

### 20. Animation Durations
```tsx
ANIMATION_DURATIONS = {
  fast: 150,
  normal: 300,
  slow: 500,
  very_slow: 1000,
}
```

### 21. Spacing
```tsx
SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
}
```

### 22. Border Radius
```tsx
BORDER_RADIUS = {
  none: 0,
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
}
```

### 23. Font Sizes
```tsx
FONT_SIZES = {
  xs: 12,
  sm: 14,
  md: 16,
  lg: 18,
  xl: 20,
  '2xl': 24,
  '3xl': 28,
  '4xl': 32,
  '5xl': 36,
}
```

### 24. Font Weights
```tsx
FONT_WEIGHTS = {
  normal: '400',
  medium: '500',
  semibold: '600',
  bold: '700',
}
```

### 25. Z-Index Layers
```tsx
Z_INDEX = {
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
}
```

### 26. Breakpoints
```tsx
BREAKPOINTS = {
  xs: 0,
  sm: 576,
  md: 768,
  lg: 992,
  xl: 1200,
  xxl: 1400,
}
```

### 27. Screen Sizes
```tsx
SCREEN_SIZES = {
  small_phone: 375,
  phone: 414,
  tablet: 768,
  large_tablet: 1024,
}
```

### 28. Platform Constants
```tsx
PLATFORMS = {
  ios: 'ios',
  android: 'android',
  web: 'web',
}
```

### 29. File Types
```tsx
FILE_TYPES = {
  image: ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
  video: ['mp4', 'webm', 'mov', 'avi', 'mkv'],
  audio: ['mp3', 'wav', 'aac', 'ogg', 'm4a'],
  document: ['pdf', 'doc', 'docx', 'txt', 'md'],
}
```

### 30. MIME Types
```tsx
MIME_TYPES = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'video/mp4': 'mp4',
  // ... más tipos
}
```

### 31. Regex Patterns
```tsx
REGEX_PATTERNS = {
  email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  url: /^https?:\/\/.+/,
  phone: /^\+?[\d\s-()]+$/,
  password: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
  uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
  hex_color: /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/,
  resolution: /^\d+x\d+$/,
}
```

### 32. Validation Rules
```tsx
VALIDATION_RULES = {
  email: { required: true, pattern: REGEX_PATTERNS.email, message: '...' },
  password: { required: true, minLength: 8, pattern: REGEX_PATTERNS.password, message: '...' },
  script: { required: true, minLength: 10, maxLength: 10000, message: '...' },
}
```

### 33. HTTP Status Codes
```tsx
HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  // ... más códigos
}
```

### 34. HTTP Methods
```tsx
HTTP_METHODS = {
  GET: 'GET',
  POST: 'POST',
  PUT: 'PUT',
  PATCH: 'PATCH',
  DELETE: 'DELETE',
}
```

### 35. Content Types
```tsx
CONTENT_TYPES = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  URL_ENCODED: 'application/x-www-form-urlencoded',
  TEXT: 'text/plain',
  HTML: 'text/html',
}
```

### 36. Feature Flags
```tsx
FEATURE_FLAGS = {
  ENABLE_BATCH_GENERATION: true,
  ENABLE_TEMPLATES: true,
  ENABLE_ANALYTICS: true,
  ENABLE_SHARING: true,
  // ... más flags
}
```

### 37. Notification Types
```tsx
NOTIFICATION_TYPES = {
  VIDEO_COMPLETED: 'video_completed',
  VIDEO_FAILED: 'video_failed',
  QUOTA_WARNING: 'quota_warning',
  SYSTEM_UPDATE: 'system_update',
}
```

### 38. Share Permissions
```tsx
SHARE_PERMISSIONS = {
  VIEW: 'view',
  EDIT: 'edit',
  ADMIN: 'admin',
}
```

### 39. Schedule Repeat Options
```tsx
SCHEDULE_REPEAT = {
  NONE: 'none',
  DAILY: 'daily',
  WEEKLY: 'weekly',
  MONTHLY: 'monthly',
}
```

### 40. Export Platforms
```tsx
EXPORT_PLATFORMS = {
  YOUTUBE: 'youtube',
  INSTAGRAM: 'instagram',
  TIKTOK: 'tiktok',
  FACEBOOK: 'facebook',
  TWITTER: 'twitter',
  LINKEDIN: 'linkedin',
}
```

### 41. Music Styles
```tsx
MUSIC_STYLES = {
  UPBEAT: 'upbeat',
  CALM: 'calm',
  EPIC: 'epic',
  CORPORATE: 'corporate',
  TECH: 'tech',
  NATURE: 'nature',
}
```

### 42. Alert Severity
```tsx
ALERT_SEVERITY = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
}
```

### 43. Storage Quotas
```tsx
STORAGE_QUOTAS = {
  FREE_TIER_VIDEOS: 10,
  FREE_TIER_STORAGE_MB: 1000,
  PREMIUM_TIER_VIDEOS: 100,
  PREMIUM_TIER_STORAGE_MB: 10000,
  ENTERPRISE_TIER_VIDEOS: -1, // Unlimited
  ENTERPRISE_TIER_STORAGE_MB: -1, // Unlimited
}
```

### 44. Rate Limits
```tsx
RATE_LIMITS = {
  FREE_TIER_PER_MINUTE: 10,
  FREE_TIER_PER_HOUR: 100,
  PREMIUM_TIER_PER_MINUTE: 60,
  PREMIUM_TIER_PER_HOUR: 1000,
  ENTERPRISE_TIER_PER_MINUTE: 300,
  ENTERPRISE_TIER_PER_HOUR: 10000,
}
```

### 45. Webhook Events
```tsx
WEBHOOK_EVENTS = {
  VIDEO_COMPLETED: 'video.completed',
  VIDEO_FAILED: 'video.failed',
  VIDEO_PROCESSING: 'video.processing',
  BATCH_COMPLETED: 'batch.completed',
}
```

### 46. Analytics Events
```tsx
ANALYTICS_EVENTS = {
  VIDEO_GENERATED: 'video_generated',
  VIDEO_DOWNLOADED: 'video_downloaded',
  VIDEO_SHARED: 'video_shared',
  TEMPLATE_USED: 'template_used',
  USER_REGISTERED: 'user_registered',
  USER_LOGGED_IN: 'user_logged_in',
  SEARCH_PERFORMED: 'search_performed',
  FEEDBACK_SUBMITTED: 'feedback_submitted',
}
```

### 47. Deep Link Paths
```tsx
DEEP_LINK_PATHS = {
  VIDEO_DETAIL: '/video/:id',
  VIDEO_GENERATION: '/video/generate',
  TEMPLATE_DETAIL: '/template/:name',
  PROFILE: '/profile',
  SETTINGS: '/settings',
  ANALYTICS: '/analytics',
}
```

### 48. Keyboard Types
```tsx
KEYBOARD_TYPES = {
  DEFAULT: 'default',
  NUMERIC: 'numeric',
  EMAIL: 'email-address',
  PHONE: 'phone-pad',
  DECIMAL: 'decimal-pad',
}
```

### 49. Return Key Types
```tsx
RETURN_KEY_TYPES = {
  DONE: 'done',
  GO: 'go',
  NEXT: 'next',
  SEARCH: 'search',
  SEND: 'send',
}
```

### 50. Text Content Types (iOS)
```tsx
TEXT_CONTENT_TYPES = {
  NONE: 'none',
  URL: 'URL',
  EMAIL: 'emailAddress',
  PASSWORD: 'password',
  NEW_PASSWORD: 'newPassword',
  ONE_TIME_CODE: 'oneTimeCode',
  // ... más tipos
}
```

### 51. Image Quality
```tsx
IMAGE_QUALITY = {
  LOW: 0.3,
  MEDIUM: 0.6,
  HIGH: 0.8,
  VERY_HIGH: 1.0,
}
```

### 52. Video Quality Presets
```tsx
VIDEO_QUALITY_PRESETS = {
  LOW: { bitrate: 1000000, resolution: '720p', fps: 24 },
  MEDIUM: { bitrate: 2500000, resolution: '1080p', fps: 30 },
  HIGH: { bitrate: 5000000, resolution: '1080p', fps: 60 },
  ULTRA: { bitrate: 10000000, resolution: '2160p', fps: 60 },
}
```

### 53. Animation Easing
```tsx
ANIMATION_EASING = {
  LINEAR: 'linear',
  EASE_IN: 'ease-in',
  EASE_OUT: 'ease-out',
  EASE_IN_OUT: 'ease-in-out',
}
```

### 54. Haptic Feedback Types
```tsx
HAPTIC_TYPES = {
  LIGHT: 'light',
  MEDIUM: 'medium',
  HEAVY: 'heavy',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error',
}
```

### 55. Error Codes
```tsx
ERROR_CODES = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  AUTH_ERROR: 'AUTH_ERROR',
  PERMISSION_ERROR: 'PERMISSION_ERROR',
  QUOTA_ERROR: 'QUOTA_ERROR',
  RATE_LIMIT_ERROR: 'RATE_LIMIT_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
}
```

### 56. Loading States
```tsx
LOADING_STATES = {
  IDLE: 'idle',
  LOADING: 'loading',
  SUCCESS: 'success',
  ERROR: 'error',
}
```

### 57. Form States
```tsx
FORM_STATES = {
  IDLE: 'idle',
  VALIDATING: 'validating',
  SUBMITTING: 'submitting',
  SUCCESS: 'success',
  ERROR: 'error',
}
```

### 58. Network States
```tsx
NETWORK_STATES = {
  UNKNOWN: 'unknown',
  NONE: 'none',
  WIFI: 'wifi',
  CELLULAR: 'cellular',
  BLUETOOTH: 'bluetooth',
  ETHERNET: 'ethernet',
  WIMAX: 'wimax',
  VPN: 'vpn',
}
```

### 59. Theme Modes
```tsx
THEME_MODES = {
  LIGHT: 'light',
  DARK: 'dark',
  AUTO: 'auto',
}
```

### 60. Accessibility
```tsx
ACCESSIBILITY = {
  MIN_TOUCH_TARGET: 44,              // iOS
  MIN_TOUCH_TARGET_ANDROID: 48,      // Android
  MIN_FONT_SIZE: 12,
  RECOMMENDED_FONT_SIZE: 16,
  MAX_CONTRAST_RATIO: 4.5,           // WCAG AA
  ENHANCED_CONTRAST_RATIO: 7,        // WCAG AAA
}
```

### 61. Performance Thresholds
```tsx
PERFORMANCE_THRESHOLDS = {
  SLOW_RENDER_MS: 16,                 // One frame at 60fps
  SLOW_NETWORK_MS: 1000,
  LARGE_BUNDLE_SIZE_KB: 500,
  MAX_IMAGE_SIZE_MB: 10,
  MAX_VIDEO_SIZE_MB: 500,
}
```

### 62. Retry Configuration
```tsx
RETRY_CONFIG = {
  MAX_RETRIES: 3,
  INITIAL_DELAY: 1000,
  MAX_DELAY: 10000,
  BACKOFF_MULTIPLIER: 2,
}
```

### 63. Pagination
```tsx
PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
  DEFAULT_PAGE: 1,
}
```

### 64. Search Configuration
```tsx
SEARCH_CONFIG = {
  MIN_QUERY_LENGTH: 1,
  MAX_QUERY_LENGTH: 100,
  DEFAULT_LIMIT: 50,
  MAX_LIMIT: 200,
  SUGGESTIONS_LIMIT: 5,
}
```

### 65. Toast Configuration
```tsx
TOAST_CONFIG = {
  DEFAULT_DURATION: 3000,
  SUCCESS_DURATION: 2000,
  ERROR_DURATION: 5000,
  WARNING_DURATION: 4000,
  INFO_DURATION: 3000,
}
```

## 🎯 Uso Recomendado

### Importar constantes específicas
```tsx
import { VIDEO_STATUS_COLORS, LIMITS, SPACING } from '@/utils/constants';
```

### Importar todas las constantes
```tsx
import * as Constants from '@/utils/constants';
```

### Usar con TypeScript types
```tsx
import type { VideoStatus, VideoStyle, AudioVoice } from '@/utils/constants';

function getStatusColor(status: VideoStatus): string {
  return VIDEO_STATUS_COLORS[status];
}
```

## 📝 Notas

- Todas las constantes usan `as const` para type safety
- Los tipos están exportados para mejor soporte TypeScript
- Las constantes están organizadas por categoría
- Valores están documentados con comentarios
- Fácil de mantener y extender

## 🔄 Actualización

Para agregar nuevas constantes:
1. Agregar en la categoría apropiada
2. Usar `as const` para type safety
3. Agregar tipos si es necesario
4. Documentar en este archivo

---

**Total de constantes**: 65+ categorías  
**Type Safety**: 100% TypeScript  
**Organización**: Por categoría lógica

