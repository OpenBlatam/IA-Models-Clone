/**
 * Core utilities module
 * Essential data manipulation utilities grouped together
 */

export * as Array from "../array";
export * as Object from "../object";
export * as String from "../string";
export * as Type from "../typeGuards";

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
} from "../array";

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
} from "../object";

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
} from "../string";

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
} from "../typeGuards";





