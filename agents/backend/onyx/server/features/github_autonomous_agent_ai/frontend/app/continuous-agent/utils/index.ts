/**
 * Barrel exports for Continuous Agent utilities
 * 
 * ## Importación Modular (Recomendado)
 * 
 * Para mejor tree-shaking y organización, considera usar la estructura modular:
 * ```ts
 * // Importación por módulo (recomendado para múltiples funciones)
 * import { Array, Object, Async } from '@/app/continuous-agent/utils/modules';
 * 
 * // Importación selectiva (recomendado para funciones individuales)
 * import { unique, deepMerge, delay } from '@/app/continuous-agent/utils/modules/core';
 * ```
 * 
 * Ver `utils/modules/README.md` para más información sobre la estructura modular.
 */

export { cn } from "./classNames";
export {
  formatRelativeTime,
  formatDateTime,
  formatDate,
  formatTime,
} from "./dateUtils";
export {
  formatNextExecutionDate,
  formatAgentStatus,
  getStatusClass,
} from "./agent-formatting";
export {
  getValidationBadgeStyle,
  getValidationBadgeTitle,
  getValidationBadgeText,
  getValidationBadgeIcon,
} from "./badge-styles";
export {
  truncateText,
  shouldTruncate,
  formatCharacterCount,
} from "./text-utils";
export {
  compareAgentCardProps,
  compareAgentDashboardProps,
} from "./memo-comparison";
// Validation utilities
export {
  validateName,
  validateDescription,
  validateFrequency,
  validateJSON,
  validateRequired,
  parseJSON,
  VALIDATION_LIMITS,
} from "./validation";
export * from "./validation/zod-schemas";
export * from "./validation/zod-validator";

// Error handling
export { getApiErrorMessage, handleApiError } from "./apiError";
export * from "./errors/agent-errors";

// Formatting utilities (consolidated)
export {
  formatNumber,
  formatCredits,
  getCreditsStatusClass,
  getCreditsStatusAriaLabel,
} from "./formatting";
export { formatFrequency, formatJSONError, getJSONErrorPosition } from "./formatters";

// Performance utilities
export * from "./performance";
export { throttle, requestIdleCallback, cancelIdleCallback, measurePerformance } from "./performance";
export { debounce } from "./debounce";

// Accessibility utilities
export * from "./accessibility";

// Type utilities
export * from "./types";
// Constants
export * from "./constants";

// Monitoring utilities
export * from "./monitoring";

// Async utilities
export * from "./async";
export {
  delay,
  retry,
  timeout,
  withTimeout,
  sleep,
  waitFor,
  type RetryOptions,
} from "./async";

// Observable utilities
export * from "./observable";

// Data transformers
export * from "./transformers";

// Comparison utilities
export * from "./comparison";

// Serialization utilities
export * from "./serialization";

// State utilities
export * from "./state";

// Security utilities
export * from "./security";

// Sanitization utilities
export * from "./sanitization";
export {
  delay,
  retry,
  timeout,
  withTimeout,
  sleep,
  waitFor,
  type RetryOptions,
} from "./async";
export {
  getStorageItem,
  setStorageItem,
  removeStorageItem,
  clearStorage,
  getStorageKeys,
  hasStorageItem,
  getStorageSize,
} from "./storage";
export {
  chunk,
  groupBy,
  unique,
  uniqueBy,
  flatten,
  flattenDeep,
  partition,
  sortBy,
  sortByDesc,
  take,
  takeWhile,
  drop,
  dropWhile,
  zip,
  unzip,
  intersection,
  difference,
  union,
} from "./array";
export {
  pick,
  omit,
  deepMerge,
  keys,
  values,
  entries,
  fromEntries,
  mapValues,
  mapKeys,
  invert,
  compact,
  defaults,
} from "./object";
export {
  truncate,
  truncateWords,
  capitalize,
  capitalizeWords,
  camelCase,
  kebabCase,
  snakeCase,
  pascalCase,
  slugify,
  padStart,
  padEnd,
  removeAccents,
  escapeHtml,
  unescapeHtml,
  stripHtml,
  wordCount,
  charCount,
  lineCount,
  normalizeWhitespace,
  removeWhitespace,
  wrap,
} from "./string";
export {
  isString,
  isNumber,
  isBoolean,
  isObject,
  isArray,
  isFunction,
  isNull,
  isUndefined,
  isNullish,
  isDefined,
  isDate,
  isError,
  isPromise,
  isEmpty as isEmptyValue,
  isNotEmpty,
  isPositive,
  isNegative,
  isInteger,
  isFloat,
  isEmail,
  isUrl,
  isJson,
} from "./typeGuards";
export {
  allSettled,
  promiseRace,
  promiseAny,
  promiseTimeout,
  promiseDelay,
  createDeferred,
  promiseRetry,
  map,
  mapSeries,
  filter,
  reduce,
  tap,
  catchError,
} from "./promise";
export {
  clamp,
  lerp,
  mapRange,
  round,
  floor,
  ceil,
  random,
  randomInt,
  randomItem,
  randomItems,
  percentage,
  percentageOf,
  average,
  sum,
  min,
  max,
  median,
  distance,
  isEven,
  isOdd,
  isBetween,
} from "./math";
export {
  parseUrl,
  buildUrl,
  parseQueryString,
  buildQueryString,
  getQueryParam,
  setQueryParam,
  removeQueryParam,
  isValidUrl,
  getDomain,
  getPath,
  getProtocol,
  isAbsoluteUrl,
  isRelativeUrl,
  joinPaths,
  normalizeUrl,
} from "./url";
export {
  compose,
  pipe,
  curry,
  memoize,
  once,
  debounceFn,
  throttleFn,
  guardArgs,
  defaultArgs,
  tap as tapFn,
  identity,
  constant,
  noop,
} from "./functional";
export {
  randomString,
  uuid,
  generateId,
  hashString,
  base64Encode,
  base64Decode,
  encodeBase64Url,
  decodeBase64Url,
  randomHex,
  randomBytes,
} from "./crypto";
export {
  first,
  last,
  findIndex,
  findLast,
  findLastIndex,
  shuffle,
  range,
  repeat,
  countBy,
  keyBy,
  sample,
  sampleSize,
  compact as compactCollection,
  uniq as uniqCollection,
  uniqBy as uniqByCollection,
  flatten as flattenCollection,
  flattenDeep as flattenDeepCollection,
  fromPairs,
  toPairs,
  isEmpty,
  size,
} from "./collection";

