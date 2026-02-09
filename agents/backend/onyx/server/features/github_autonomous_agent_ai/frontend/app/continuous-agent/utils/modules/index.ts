/**
 * Modular exports for Continuous Agent utilities
 * 
 * This file provides organized module exports for better tree-shaking
 * and code organization. Import only what you need.
 * 
 * ## Recommended Import Patterns
 * 
 * ### By Specific Module (Best for tree-shaking)
 * ```ts
 * import { unique, deepClone, camelCase } from '@/app/continuous-agent/utils/modules/core';
 * import { delay, retry, debounce } from '@/app/continuous-agent/utils/modules/async';
 * import { formatDate, formatNumber } from '@/app/continuous-agent/utils/modules/format';
 * ```
 * 
 * ### By Module Namespace (When using many functions from same module)
 * ```ts
 * import { Array, Object, String } from '@/app/continuous-agent/utils/modules/core';
 * const unique = Array.unique([1, 2, 2]);
 * const merged = Object.deepMerge({ a: 1 }, { b: 2 });
 * ```
 * 
 * ### From Main Index (All categories available)
 * ```ts
 * import { Array, Object, Async, Format } from '@/app/continuous-agent/utils/modules';
 * ```
 * 
 * See `modules/README.md` for complete guide.
 */

export * as Array from "../array";
export * as Object from "../object";
export * as String from "../string";
export * as Type from "../typeGuards";
export * as Async from "../async";
export * as Promise from "../promise";
export * as Performance from "../performance";
export * as Date from "../dateUtils";
export * as Formatting from "../formatting";
export * as Formatters from "../formatters";
export * as Validation from "../validation";
export * as ApiError from "../apiError";
export * as Storage from "../storage";
export * as Math from "../math";
export * as Url from "../url";
export * as Functional from "../functional";
export * as Crypto from "../crypto";
export * as Collection from "../collection";

export * from "./core";
export * from "./async";
export * from "./format";
export * from "./validation";
export * from "./storage";
export * from "./ui";
export * from "./math";
export * from "./url";
export * from "./functional";
export * from "./crypto";
export * from "./collection";

export { cn } from "./ui";

