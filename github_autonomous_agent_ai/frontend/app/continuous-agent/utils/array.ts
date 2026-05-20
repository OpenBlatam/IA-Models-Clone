/**
 * Array manipulation utilities
 * Uses lodash-es for optimal performance and tree-shaking
 */

import chunk from "lodash-es/chunk";
import groupBy from "lodash-es/groupBy";
import uniq from "lodash-es/uniq";
import uniqBy from "lodash-es/uniqBy";
import flatten from "lodash-es/flatten";
import flattenDeep from "lodash-es/flattenDeep";
import partition from "lodash-es/partition";
import sortBy from "lodash-es/sortBy";
import take from "lodash-es/take";
import takeWhile from "lodash-es/takeWhile";
import drop from "lodash-es/drop";
import dropWhile from "lodash-es/dropWhile";
import zip from "lodash-es/zip";
import intersection from "lodash-es/intersection";
import difference from "lodash-es/difference";
import union from "lodash-es/union";
import orderBy from "lodash-es/orderBy";

export { chunk, groupBy, flatten, flattenDeep, partition, take, drop, zip, intersection, difference, union };

export const unique = <T>(array: readonly T[]): T[] => {
  return uniq(array);
};

export const uniqueBy = <T, K>(
  array: readonly T[],
  keyFn: (item: T) => K
): T[] => {
  return uniqBy(array, keyFn as (item: T) => unknown);
};

export { sortBy };

export const sortByDesc = <T>(
  array: readonly T[],
  keyFn: (item: T) => number | string
): T[] => {
  return orderBy(array, [keyFn as (item: T) => unknown], ["desc"]);
};

export { takeWhile, dropWhile };

export const unzip = <T, U>(zipped: readonly [T, U][]): [T[], U[]] => {
  const array1: T[] = [];
  const array2: U[] = [];

  for (const [item1, item2] of zipped) {
    array1.push(item1);
    array2.push(item2);
  }

  return [array1, array2];
};

