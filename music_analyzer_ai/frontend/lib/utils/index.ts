/**
 * Utility functions exports.
 * Centralized export point for all utility functions.
 */

export {
  cn,
  formatDuration,
  formatBPM,
  formatPercentage,
  debounce,
} from '../utils';

export {
  throttle,
  rafThrottle,
  isBrowser,
  isProduction,
  measurePerformance,
} from './performance';

export {
  safeParse,
  validateData,
  validateField,
  createValidator,
  combineValidationResults,
  validateAndTransform,
} from './validation';

export {
  sanitizeString,
  sanitizeSearchQuery,
  sanitizeUrl,
  escapeHtml,
  isValidEmail,
  isValidUrl,
} from './sanitization';

export {
  formatNumber,
  formatDate,
  formatRelativeTime,
  formatFileSize,
  truncateText,
  capitalize,
  formatPercent,
} from './formatting';

export {
  unique,
  groupBy,
  chunk,
  flatten,
  sortBy,
  partition,
  intersection,
  difference,
} from './array';

export {
  deepMerge,
  pick,
  omit,
  isEmpty,
  get,
  set,
  fromEntries,
  isEqual,
} from './object';

export {
  toCamelCase,
  toKebabCase,
  toSnakeCase,
  toPascalCase,
  capitalizeFirst,
  trim,
  removeWhitespace,
  randomString,
  startsWith,
  endsWith,
  replaceAll,
} from './string';

export {
  delay,
  timeout,
  withTimeout,
  retry,
  debounceAsync,
  throttleAsync,
  pLimit,
  cancellable,
} from './async';

export { localStorage, sessionStorage } from './storage';

export {
  formatDate as formatDateLocale,
  formatRelativeDate,
  formatSmartDate,
  startOfDay,
  endOfDay,
  isPast,
  isFuture,
  addDays,
  differenceInDays,
} from './date';

export {
  parseQueryParams,
  buildUrl,
  getQueryParam,
  setQueryParam,
  removeQueryParam,
  getBaseUrl,
  isAbsoluteUrl,
  toAbsoluteUrl,
  getDomain,
  getPath,
} from './url';

export {
  hexToRgb,
  rgbToHex,
  rgbToHsl,
  lightenColor,
  darkenColor,
  getContrastRatio,
  isDarkColor,
  getContrastColor,
} from './color';

export {
  clamp,
  isInRange,
  random,
  roundTo,
  formatNumber as formatNumberLocale,
  parseNumber,
  isValidNumber,
  formatBytes,
  calculatePercentage,
  lerp,
} from './number';

export { copyToClipboard, readFromClipboard } from './clipboard';

export {
  formatTime,
  formatTimeLong,
  parseTime,
  getTimestamp,
  getTimestampSeconds,
  msToSeconds,
  secondsToMs,
  getTimeDifference,
  isPastTime,
  isFutureTime,
} from './time';

export { logger, createLogger } from './logger';
export {
  handleError,
  createErrorHandler,
  withErrorHandling,
} from './error-handler';
export type { ErrorHandlerOptions } from './error-handler';
export { retry, retryWithCondition } from './retry';
export type { RetryOptions } from './retry';
export {
  setCookie,
  getCookie,
  removeCookie,
  getAllCookies,
} from './cookie';

export {
  querySelector,
  querySelectorAll,
  matches,
  closest,
  scrollIntoView,
  getBoundingClientRect,
  isElementVisible,
  getComputedStyles,
  addClass,
  removeClass,
  toggleClass,
  hasClass,
} from './dom';

export {
  isMobile,
  isTablet,
  isDesktop,
  getDeviceType,
  isTouchDevice,
  getUserAgent,
  getPlatform,
  isIOS,
  isAndroid,
  isWindows,
  isMac,
  isLinux,
  getBrowser,
} from './device';

export {
  easing,
  animate,
  fadeIn,
  fadeOut,
  slideIn,
  slideOut,
} from './animation';

export {
  setWithExpiration,
  getWithExpiration,
  clearExpired,
  getStorageSize,
  getStorageQuota,
} from './storage-advanced';

export {
  formatCurrency,
  formatPercent,
  formatNumber as formatNumberAdvanced,
  formatDateRange,
  formatPhone,
  formatCardNumber,
  formatInitials,
  formatSlug,
  formatFileSize as formatFileSizeAdvanced,
} from './formatting-advanced';

export {
  isValidEmail as isValidEmailAdvanced,
  isValidUrl as isValidUrlAdvanced,
  isValidPhone,
  isValidCardNumber,
  isValidDate,
  isValidTime,
  validatePassword,
  isValidCardExpiry,
} from './validation-advanced';

export {
  searchText,
  fuzzySearch,
  highlightSearch,
  filterBySearch,
  sortByRelevance,
  getSearchSuggestions,
} from './search';

export {
  calculatePagination,
  getPageItems,
  generatePageNumbers,
} from './pagination';
export type {
  PaginationOptions,
  PaginationResult,
} from './pagination';

export {
  compare,
  compareIgnoreCase,
  compareNumbers,
  compareDates,
  isEqual as isEqualDeep,
  isGreaterThan,
  isLessThan,
  isBetween,
} from './comparison';

export {
  sortBy,
  sortByDesc,
  sortByMultiple,
  sortWith,
  reverse as reverseArray,
  shuffle,
} from './sorting';

export {
  filter as filterArray,
  compact,
  uniqueBy,
  filterByAll,
  filterByAny,
  filterByValue,
  filterByRange,
} from './filtering';

export {
  map as mapArray,
  reduce as reduceArray,
  groupBy as groupByArray,
  partition as partitionArray,
  pick as pickObject,
  omit as omitObject,
  transformKeys,
  transformValues,
} from './transform';

export {
  sum,
  average,
  min,
  max,
  count,
  median,
  mode,
} from './aggregation';

export { Queue, Stack } from './queue';
export { Cache, LRUCache } from './cache';
export { EventEmitter } from './event-emitter';
export type { EventListener } from './event-emitter';
export {
  createPromise,
  delayPromise,
  timeoutPromise,
  sequence,
  parallel,
  retryPromise,
  debouncePromise,
} from './promise';
export { Observable } from './observable';
export type { Observer } from './observable';
export {
  createWorker,
  createWorkerFromString,
  createWorkerFromURL,
  terminateWorker,
  sendMessage,
} from './worker';
export {
  hash,
  hashObject,
  hashString,
  hashObjectString,
  randomHash,
} from './hash';
export {
  generateId,
  generateUUID,
  generateShortId,
  generateNumericId,
} from './id';
export {
  encodeBase64,
  decodeBase64,
  encodeObjectBase64,
  decodeObjectBase64,
  encodeBase64URL,
  decodeBase64URL,
} from './encoding';
export {
  compressRLE,
  decompressRLE,
  compressJSON,
  estimateCompressionRatio,
} from './compression';
export {
  diffObjects,
  diffArrays,
  applyDiff,
} from './diff';
export type { DiffOperation, DiffResult } from './diff';
export {
  memoize,
  memoizeLRU,
  memoizeTTL,
} from './memoization';
export type { MemoizedFunction } from './memoization';
export {
  compose,
  pipe,
  curry,
  partial,
  negate,
  all,
  any,
  constant,
  identity,
  noop,
} from './functional';
export {
  range,
  infinite,
  repeat,
  cycle,
  mapIterator,
  filterIterator,
  takeIterator,
  skipIterator,
  zip,
  iteratorToArray,
} from './iterators';
export { batch, batchAsync, batchRAF } from './batch';
export { RateLimiter, createRateLimiter } from './rate-limit';
export { PriorityQueue, CircularQueue } from './queue-advanced';
export { Semaphore, createSemaphore } from './semaphore';
export { Stream, stream } from './stream';
export { Reactive, reactive, computed } from './reactive';
export {
  reactiveProxy,
  readonlyProxy,
  validationProxy,
  loggingProxy,
} from './proxy-utils';
export {
  getPropertyNames,
  getPropertyDescriptors,
  hasProperty,
  getProperty,
  setProperty,
  deleteProperty,
  defineProperty,
  getPrototype,
  setPrototype,
} from './reflection';
export {
  StateMachine,
  createStateMachine,
} from './state-machine';
export type {
  StateTransition,
  StateMachineConfig,
} from './state-machine';
export { Pipeline, pipeline } from './pipeline';
export type { PipelineStage } from './pipeline';
export {
  MiddlewareChain,
  createMiddlewareChain,
} from './middleware';
export type { Middleware } from './middleware';
export { Chain, chain } from './chain';
export {
  PerformanceMonitor,
  performanceMonitor,
  measurePerformance,
  measurePerformanceAsync,
} from './performance-monitor';
export type { PerformanceMetric } from './performance-monitor';
export {
  AnalyticsTracker,
  analytics,
  trackEvent,
  trackPageView,
  trackAction,
  trackError,
} from './analytics';
export type { AnalyticsEvent } from './analytics';
export {
  retryAdvanced,
  createRetryFunction,
} from './retry-advanced';
export type {
  RetryStrategy,
  AdvancedRetryOptions,
} from './retry-advanced';
export {
  CircuitBreaker,
  createCircuitBreaker,
} from './circuit-breaker';
export type {
  CircuitBreakerState,
  CircuitBreakerOptions,
} from './circuit-breaker';
export {
  withTimeout,
  createTimeoutPromise,
  withRetryAndTimeout,
} from './timeout-advanced';
export type { TimeoutOptions } from './timeout-advanced';

export {
  memoizeComponent,
  lazyLoadComponent,
  conditionalRender,
  clientOnly,
} from './react-optimization';
export type {
  MemoizeOptions,
  LazyLoadOptions,
} from './react-optimization';

export {
  getErrorSeverity,
  isRecoverableError,
  getUserFriendlyMessage,
  handleError,
  createErrorHandler,
  withErrorHandling,
  retryWithBackoff,
  recoverFromError,
} from './error-handling-advanced';
export type {
  ErrorSeverity,
  ErrorContext,
  ErrorReport,
  ErrorHandlingOptions,
  RecoveryStrategy,
} from './error-handling-advanced';

export {
  createAriaLiveRegion,
  announceToScreenReader,
  getAccessibleLabel,
  isFocusable,
  getFocusableElements,
  trapFocus,
  restoreFocus,
  saveFocus,
  prefersReducedMotion,
  prefersHighContrast,
  getContrastRatio,
  meetsContrastRatio,
  validateAriaAttributes,
} from './accessibility';
export type {
  AriaLivePriority,
  AriaLiveRegionOptions,
} from './accessibility';

export {
  sanitizeXss,
  sanitizeUrl,
  generateSecureToken,
  hashString,
  validateCSP,
  containsDangerousContent,
  validateFileType,
  validateFileSize,
  createSecureDownload,
  ClientRateLimiter,
} from './security-advanced';
export type {
  XssSanitizeOptions,
} from './security-advanced';

