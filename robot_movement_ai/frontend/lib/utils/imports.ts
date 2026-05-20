/**
 * Import utilities for better code organization
 */

// Re-export commonly used utilities for convenience
export {
  // Core
  cn,
  logger,
  cache,
  enhancedStorage,
  
  // Constants
  API_CONFIG,
  ROBOT_CONFIG,
  UI_CONFIG,
  STORAGE_KEYS,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  
  // Formatting
  formatNumber,
  formatCurrency,
  formatDate,
  formatDuration,
  formatFileSize,
  formatPercentage,
  formatPosition,
  
  // Validation
  safeParse,
  validateForm,
  
  // Robot
  validatePosition,
  clampPosition,
  calculateDistance,
  formatPosition as formatRobotPosition,
  
  // Type guards
  isDefined,
  isString,
  isNumber,
  isObject,
  isArray,
  
  // Safe operations
  safeJsonParse,
  safeNumberParse,
  safeGet,
  safeCall,
} from './index';



