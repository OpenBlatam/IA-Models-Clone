/**
 * String manipulation utilities
 * Uses lodash-es for optimal performance and tree-shaking where applicable
 */

import camelCase from "lodash-es/camelCase";
import kebabCase from "lodash-es/kebabCase";
import snakeCase from "lodash-es/snakeCase";
import capitalize from "lodash-es/capitalize";
import padStart from "lodash-es/padStart";
import padEnd from "lodash-es/padEnd";
import lodashTruncate from "lodash-es/truncate";
import escape from "lodash-es/escape";
import unescape from "lodash-es/unescape";
import startCase from "lodash-es/startCase";
import deburr from "lodash-es/deburr";

export { camelCase, kebabCase, snakeCase, capitalize, padStart, padEnd };

export const truncate = (
  str: string,
  length: number,
  suffix = "..."
): string => {
  return lodashTruncate(str, { length, omission: suffix });
};

export const truncateWords = (
  str: string,
  wordCount: number,
  suffix = "..."
): string => {
  const words = str.split(/\s+/);
  if (words.length <= wordCount) {
    return str;
  }
  return words.slice(0, wordCount).join(" ") + suffix;
};

export const capitalizeWords = (str: string): string => {
  return startCase(str);
};

export const pascalCase = (str: string): string => {
  return startCase(str).replace(/\s+/g, "");
};

export const slugify = (str: string): string => {
  return deburr(str)
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
};

export const removeAccents = (str: string): string => {
  return deburr(str);
};

export const escapeHtml = (str: string): string => {
  return escape(str);
};

export const unescapeHtml = (str: string): string => {
  return unescape(str);
};

export const stripHtml = (str: string): string => {
  return str.replace(/<[^>]*>/g, "");
};

export const wordCount = (str: string): number => {
  return str.trim().split(/\s+/).filter(Boolean).length;
};

export const charCount = (str: string): number => {
  return str.length;
};

export const lineCount = (str: string): number => {
  return str.split(/\n/).length;
};

export const normalizeWhitespace = (str: string): string => {
  return str.replace(/\s+/g, " ").trim();
};

export const removeWhitespace = (str: string): string => {
  return str.replace(/\s+/g, "");
};

export const wrap = (
  str: string,
  width: number,
  breakChar = "\n"
): string => {
  const words = str.split(" ");
  const lines: string[] = [];
  let currentLine = "";

  for (const word of words) {
    const testLine = currentLine ? `${currentLine} ${word}` : word;
    if (testLine.length <= width) {
      currentLine = testLine;
    } else {
      if (currentLine) {
        lines.push(currentLine);
      }
      currentLine = word;
    }
  }

  if (currentLine) {
    lines.push(currentLine);
  }

  return lines.join(breakChar);
};

