/**
 * Type guard utilities
 * Uses lodash-es for optimal performance and tree-shaking where applicable
 */

import isString from "lodash-es/isString";
import isNumber from "lodash-es/isNumber";
import isBoolean from "lodash-es/isBoolean";
import isObject from "lodash-es/isObject";
import isArray from "lodash-es/isArray";
import isFunction from "lodash-es/isFunction";
import isNull from "lodash-es/isNull";
import isUndefined from "lodash-es/isUndefined";
import isDate from "lodash-es/isDate";
import isError from "lodash-es/isError";
import isEmpty from "lodash-es/isEmpty";
import isNil from "lodash-es/isNil";
import isInteger from "lodash-es/isInteger";

export {
  isString,
  isNumber,
  isBoolean,
  isObject,
  isArray,
  isFunction,
  isNull,
  isUndefined,
  isDate,
  isError,
  isEmpty,
};

export const isNullish = (value: unknown): value is null | undefined => {
  return isNil(value);
};

export const isDefined = <T>(value: T | null | undefined): value is T => {
  return !isNil(value);
};

export const isPromise = <T = unknown>(
  value: unknown
): value is Promise<T> => {
  return (
    typeof value === "object" &&
    value !== null &&
    "then" in value &&
    typeof (value as Promise<T>).then === "function"
  );
};

export const isNotEmpty = (value: unknown): boolean => {
  return !isEmpty(value);
};

export const isPositive = (value: number): boolean => {
  return isNumber(value) && value > 0;
};

export const isNegative = (value: number): boolean => {
  return isNumber(value) && value < 0;
};

export const isInteger = (value: unknown): value is number => {
  return isNumber(value) && Number.isInteger(value);
};

export const isFloat = (value: unknown): value is number => {
  return isNumber(value) && !Number.isInteger(value);
};

export const isEmail = (value: string): boolean => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
};

export const isUrl = (value: string): boolean => {
  try {
    new URL(value);
    return true;
  } catch {
    return false;
  }
};

export const isJson = (value: string): boolean => {
  try {
    JSON.parse(value);
    return true;
  } catch {
    return false;
  }
};

