/**
 * Number manipulation utilities
 */

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function random(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

export function round(value: number, decimals = 0): number {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

export function floor(value: number, decimals = 0): number {
  const factor = Math.pow(10, decimals);
  return Math.floor(value * factor) / factor;
}

export function ceil(value: number, decimals = 0): number {
  const factor = Math.pow(10, decimals);
  return Math.ceil(value * factor) / factor;
}

export function padStart(value: number, length: number, fillString = '0'): string {
  return String(value).padStart(length, fillString);
}

export function padEnd(value: number, length: number, fillString = '0'): string {
  return String(value).padEnd(length, fillString);
}

export function formatNumber(
  value: number,
  options?: {
    decimals?: number;
    thousandSeparator?: string;
    decimalSeparator?: string;
    prefix?: string;
    suffix?: string;
  }
): string {
  const {
    decimals = 2,
    thousandSeparator = ',',
    decimalSeparator = '.',
    prefix = '',
    suffix = '',
  } = options || {};

  const fixed = value.toFixed(decimals);
  const [integer, decimal] = fixed.split('.');

  const formattedInteger = integer.replace(/\B(?=(\d{3})+(?!\d))/g, thousandSeparator);
  const formattedDecimal = decimal ? `${decimalSeparator}${decimal}` : '';

  return `${prefix}${formattedInteger}${formattedDecimal}${suffix}`;
}

export function formatCurrency(
  value: number,
  currency = 'USD',
  locale = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(value);
}

export function formatPercentage(value: number, decimals = 0): string {
  return `${round(value, decimals)}%`;
}

export function parseNumber(value: string | number, defaultValue = 0): number {
  if (typeof value === 'number') return value;
  if (typeof value !== 'string') return defaultValue;

  const cleaned = value.replace(/[^\d.-]/g, '');
  const parsed = parseFloat(cleaned);

  return isNaN(parsed) ? defaultValue : parsed;
}

export function isEven(value: number): boolean {
  return value % 2 === 0;
}

export function isOdd(value: number): boolean {
  return value % 2 !== 0;
}

export function isInteger(value: number): boolean {
  return Number.isInteger(value);
}

export function isFloat(value: number): boolean {
  return !Number.isInteger(value);
}

export function isPositive(value: number): boolean {
  return value > 0;
}

export function isNegative(value: number): boolean {
  return value < 0;
}

export function isZero(value: number): boolean {
  return value === 0;
}

export function isBetween(value: number, min: number, max: number): boolean {
  return value >= min && value <= max;
}

export function sum(numbers: number[]): number {
  return numbers.reduce((acc, num) => acc + num, 0);
}

export function average(numbers: number[]): number {
  if (numbers.length === 0) return 0;
  return sum(numbers) / numbers.length;
}

export function min(numbers: number[]): number {
  return Math.min(...numbers);
}

export function max(numbers: number[]): number {
  return Math.max(...numbers);
}

export function median(numbers: number[]): number {
  if (numbers.length === 0) return 0;

  const sorted = [...numbers].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  return sorted[middle];
}

export function mode(numbers: number[]): number | null {
  if (numbers.length === 0) return null;

  const frequency: Record<number, number> = {};
  let maxFreq = 0;
  let modeValue: number | null = null;

  numbers.forEach((num) => {
    frequency[num] = (frequency[num] || 0) + 1;
    if (frequency[num] > maxFreq) {
      maxFreq = frequency[num];
      modeValue = num;
    }
  });

  return modeValue;
}

export function variance(numbers: number[]): number {
  if (numbers.length === 0) return 0;

  const avg = average(numbers);
  const squaredDiffs = numbers.map((num) => Math.pow(num - avg, 2));
  return average(squaredDiffs);
}

export function standardDeviation(numbers: number[]): number {
  return Math.sqrt(variance(numbers));
}

export function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

export function normalize(value: number, min: number, max: number): number {
  return (value - min) / (max - min);
}

export function denormalize(normalized: number, min: number, max: number): number {
  return normalized * (max - min) + min;
}

export function toRadians(degrees: number): number {
  return (degrees * Math.PI) / 180;
}

export function toDegrees(radians: number): number {
  return (radians * 180) / Math.PI;
}


