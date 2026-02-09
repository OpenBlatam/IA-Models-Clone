/**
 * Object manipulation utilities
 * Uses lodash-es for optimal performance and tree-shaking
 */

import pick from "lodash-es/pick";
import omit from "lodash-es/omit";
import merge from "lodash-es/merge";
import keys from "lodash-es/keys";
import values from "lodash-es/values";
import mapValues from "lodash-es/mapValues";
import mapKeys from "lodash-es/mapKeys";
import invert from "lodash-es/invert";
import defaults from "lodash-es/defaults";
import pickBy from "lodash-es/pickBy";

export { pick, omit, keys, values, mapValues, mapKeys, invert, defaults };

export const deepMerge = <T extends Record<string, unknown>>(
  target: T,
  ...sources: Partial<T>[]
): T => {
  return merge({}, target, ...sources) as T;
};

export const entries = <T extends Record<string, unknown>>(
  obj: T
): [keyof T, T[keyof T]][] => {
  return Object.entries(obj) as [keyof T, T[keyof T]][];
};

export const fromEntries = <K extends string, V>(
  entries: readonly [K, V][]
): Record<K, V> => {
  return Object.fromEntries(entries) as Record<K, V>;
};

export const compact = <T extends Record<string, unknown | null | undefined>>(
  obj: T
): Partial<T> => {
  return pickBy(obj, (value) => value !== null && value !== undefined) as Partial<T>;
};

