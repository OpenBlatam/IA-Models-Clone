/**
 * Constants used across the ChatInterface
 */

// Storage Keys
export const STORAGE_KEYS = {
  CHAT_STATE: 'chat-state',
  CHAT_SETTINGS: 'chat-settings',
  CHAT_THEME: 'chat-theme',
  FAVORITES: 'bulk-chat-favorites',
  PINS: 'bulk-chat-pins',
  ARCHIVES: 'bulk-chat-archives',
  TAGS: 'bulk-chat-tags',
  NOTES: 'bulk-chat-notes',
  BOOKMARKS: 'bulk-chat-bookmarks',
  COLLABORATION: 'chat-collaboration-settings',
  NOTIFICATIONS: 'chat-notification-settings',
  ACCESSIBILITY: 'chat-accessibility-settings',
} as const

// Default Values
export const DEFAULTS = {
  VIEW_MODE: 'normal' as const,
  FONT_SIZE: 'medium' as const,
  THEME: 'dark' as const,
  AUTO_SCROLL: true,
  AUTO_SAVE: true,
  MAX_MESSAGE_LENGTH: 10000,
  CACHE_TTL: 3600000, // 1 hour
  DEBOUNCE_DELAY: 300,
  THROTTLE_DELAY: 100,
} as const

// Export Formats
export const EXPORT_FORMATS = [
  'json',
  'txt',
  'md',
  'html',
  'csv',
  'xml',
  'yaml',
  'pdf',
] as const

// View Modes
export const VIEW_MODES = ['normal', 'compact', 'comfortable'] as const

// Font Sizes
export const FONT_SIZES = ['small', 'medium', 'large'] as const

// Themes
export const THEMES = ['dark', 'light', 'auto'] as const

// Message Roles
export const MESSAGE_ROLES = ['user', 'assistant', 'system'] as const

// Filter Roles
export const FILTER_ROLES = ['all', 'user', 'assistant', 'system'] as const

// Reading Speeds
export const READING_SPEEDS = ['slow', 'normal', 'fast'] as const

// Compression Levels
export const COMPRESSION_LEVELS = ['low', 'medium', 'high'] as const

// Notification Variants
export const NOTIFICATION_VARIANTS = ['danger', 'warning', 'info', 'success'] as const

// Modal Sizes
export const MODAL_SIZES = ['sm', 'md', 'lg', 'xl', 'full'] as const

// Performance Thresholds
export const PERFORMANCE_THRESHOLDS = {
  SLOW_RENDER: 16, // ms (60fps = 16.67ms per frame)
  VERY_SLOW_RENDER: 100, // ms
  LARGE_MESSAGE_COUNT: 1000,
  LARGE_CACHE_SIZE: 100,
} as const

// Validation Limits
export const VALIDATION_LIMITS = {
  MAX_MESSAGE_LENGTH: 10000,
  MAX_SEARCH_LENGTH: 200,
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  MAX_TAGS_PER_MESSAGE: 10,
  MAX_REACTIONS_PER_MESSAGE: 20,
} as const

// Regex Patterns
export const REGEX_PATTERNS = {
  URL: /https?:\/\/[^\s]+/g,
  EMAIL: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
  CODE_BLOCK: /```[\s\S]*?```/g,
  INLINE_CODE: /`[^`]+`/g,
  MARKDOWN_LINK: /\[([^\]]+)\]\(([^)]+)\)/g,
} as const

// Color Palette
export const COLORS = {
  PRIMARY: '#007bff',
  SECONDARY: '#6c757d',
  SUCCESS: '#28a745',
  DANGER: '#dc3545',
  WARNING: '#ffc107',
  INFO: '#17a2b8',
  DARK: '#343a40',
  LIGHT: '#f8f9fa',
} as const

// Animation Durations (ms)
export const ANIMATION_DURATIONS = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
  VERY_SLOW: 1000,
} as const

// Breakpoints
export const BREAKPOINTS = {
  MOBILE: 640,
  TABLET: 768,
  DESKTOP: 1024,
  WIDE: 1280,
} as const

// Z-Index Layers
export const Z_INDEX = {
  DROPDOWN: 1000,
  STICKY: 1020,
  FIXED: 1030,
  MODAL_BACKDROP: 1040,
  MODAL: 1050,
  POPOVER: 1060,
  TOOLTIP: 1070,
} as const

// Error Messages
export const ERROR_MESSAGES = {
  INVALID_MESSAGE: 'El mensaje no es válido',
  MESSAGE_TOO_LONG: 'El mensaje es demasiado largo',
  SEARCH_TOO_LONG: 'La búsqueda es demasiado larga',
  FILE_TOO_LARGE: 'El archivo es demasiado grande',
  INVALID_FORMAT: 'Formato no válido',
  STORAGE_QUOTA_EXCEEDED: 'No hay suficiente espacio de almacenamiento',
  NETWORK_ERROR: 'Error de red',
  PERMISSION_DENIED: 'Permiso denegado',
} as const

// Success Messages
export const SUCCESS_MESSAGES = {
  MESSAGE_SENT: 'Mensaje enviado',
  MESSAGE_SAVED: 'Mensaje guardado',
  EXPORT_SUCCESS: 'Exportación exitosa',
  IMPORT_SUCCESS: 'Importación exitosa',
  SETTINGS_SAVED: 'Configuración guardada',
} as const

// Localization Keys
export const I18N_KEYS = {
  // Common
  SAVE: 'save',
  CANCEL: 'cancel',
  DELETE: 'delete',
  EDIT: 'edit',
  CLOSE: 'close',
  CONFIRM: 'confirm',
  
  // Messages
  NO_MESSAGES: 'no_messages',
  LOADING: 'loading',
  SENDING: 'sending',
  
  // Actions
  FAVORITE: 'favorite',
  PIN: 'pin',
  ARCHIVE: 'archive',
  SHARE: 'share',
  COPY: 'copy',
  REPLY: 'reply',
  
  // Filters
  FILTER_BY_ROLE: 'filter_by_role',
  FILTER_BY_DATE: 'filter_by_date',
  FILTER_BY_CONTENT: 'filter_by_content',
  
  // Settings
  VIEW_MODE: 'view_mode',
  FONT_SIZE: 'font_size',
  THEME: 'theme',
  AUTO_SCROLL: 'auto_scroll',
} as const




