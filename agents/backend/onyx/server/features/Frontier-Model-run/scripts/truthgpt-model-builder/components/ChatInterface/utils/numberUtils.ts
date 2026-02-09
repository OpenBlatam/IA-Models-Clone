/**
 * Number utility functions
 */

/**
 * Clamp number between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/**
 * Round to decimal places
 */
export function round(value: number, decimals: number = 0): number {
  const factor = Math.pow(10, decimals)
  return Math.round(value * factor) / factor
}

/**
 * Format number with commas
 */
export function formatNumber(value: number, locale: string = 'es-ES'): string {
  return new Intl.NumberFormat(locale).format(value)
}

/**
 * Format currency
 */
export function formatCurrency(
  value: number,
  currency: string = 'USD',
  locale: string = 'es-ES'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(value)
}

/**
 * Format percentage
 */
export function formatPercentage(
  value: number,
  total: number = 100,
  decimals: number = 1
): string {
  if (total === 0) return '0%'
  const percentage = (value / total) * 100
  return `${round(percentage, decimals)}%`
}

/**
 * Generate random number
 */
export function random(min: number = 0, max: number = 1): number {
  return Math.random() * (max - min) + min
}

/**
 * Generate random integer
 */
export function randomInt(min: number, max: number): number {
  return Math.floor(random(min, max + 1))
}

/**
 * Check if number is between range
 */
export function isBetween(value: number, min: number, max: number): boolean {
  return value >= min && value <= max
}

/**
 * Check if number is even
 */
export function isEven(value: number): boolean {
  return value % 2 === 0
}

/**
 * Check if number is odd
 */
export function isOdd(value: number): boolean {
  return value % 2 !== 0
}

/**
 * Check if number is integer
 */
export function isInteger(value: number): boolean {
  return Number.isInteger(value)
}

/**
 * Check if number is float
 */
export function isFloat(value: number): boolean {
  return !isInteger(value)
}

/**
 * Check if number is positive
 */
export function isPositive(value: number): boolean {
  return value > 0
}

/**
 * Check if number is negative
 */
export function isNegative(value: number): boolean {
  return value < 0
}

/**
 * Check if number is zero
 */
export function isZero(value: number): boolean {
  return value === 0
}

/**
 * Convert bytes to human readable
 */
export function bytesToSize(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return round(bytes / Math.pow(k, i), decimals) + ' ' + sizes[i]
}

/**
 * Convert size to bytes
 */
export function sizeToBytes(size: string): number {
  const units: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
    TB: 1024 * 1024 * 1024 * 1024,
  }

  const match = size.match(/^(\d+(?:\.\d+)?)\s*([A-Z]+)$/i)
  if (!match) return 0

  const value = parseFloat(match[1])
  const unit = match[2].toUpperCase()

  return value * (units[unit] || 1)
}

/**
 * Calculate percentage
 */
export function calculatePercentage(value: number, total: number): number {
  if (total === 0) return 0
  return (value / total) * 100
}

/**
 * Calculate percentage change
 */
export function percentageChange(oldValue: number, newValue: number): number {
  if (oldValue === 0) return newValue > 0 ? 100 : 0
  return ((newValue - oldValue) / oldValue) * 100
}

/**
 * Linear interpolation
 */
export function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * clamp(t, 0, 1)
}

/**
 * Map value from one range to another
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin
}

/**
 * Get min value from array
 */
export function min(values: number[]): number {
  return Math.min(...values)
}

/**
 * Get max value from array
 */
export function max(values: number[]): number {
  return Math.max(...values)
}

/**
 * Get average from array
 */
export function average(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((sum, val) => sum + val, 0) / values.length
}

/**
 * Get sum from array
 */
export function sum(values: number[]): number {
  return values.reduce((sum, val) => sum + val, 0)
}

/**
 * Get median from array
 */
export function median(values: number[]): number {
  if (values.length === 0) return 0

  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)

  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid]
}

/**
 * Get standard deviation
 */
export function standardDeviation(values: number[]): number {
  if (values.length === 0) return 0

  const avg = average(values)
  const squareDiffs = values.map(value => Math.pow(value - avg, 2))
  const avgSquareDiff = average(squareDiffs)

  return Math.sqrt(avgSquareDiff)
}




