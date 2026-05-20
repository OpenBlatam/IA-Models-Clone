/**
 * Math utilities
 */

// Clamp value between min and max
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

// Linear interpolation
export function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

// Map value from one range to another
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

// Check if number is in range
export function inRange(value: number, min: number, max: number): boolean {
  return value >= min && value <= max;
}

// Round to decimal places
export function round(value: number, decimals: number = 0): number {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

// Calculate percentage
export function percentage(value: number, total: number): number {
  if (total === 0) return 0;
  return (value / total) * 100;
}

// Calculate distance between two points
export function distance(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
}

// Calculate distance in 3D
export function distance3D(
  x1: number,
  y1: number,
  z1: number,
  x2: number,
  y2: number,
  z2: number
): number {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) + Math.pow(z2 - z1, 2));
}

// Generate random number in range
export function random(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

// Generate random integer in range
export function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Check if number is even
export function isEven(n: number): boolean {
  return n % 2 === 0;
}

// Check if number is odd
export function isOdd(n: number): boolean {
  return n % 2 !== 0;
}

// Calculate average
export function average(numbers: number[]): number {
  if (numbers.length === 0) return 0;
  return numbers.reduce((sum, n) => sum + n, 0) / numbers.length;
}

// Calculate sum
export function sum(numbers: number[]): number {
  return numbers.reduce((sum, n) => sum + n, 0);
}

// Calculate min
export function min(numbers: number[]): number {
  return Math.min(...numbers);
}

// Calculate max
export function max(numbers: number[]): number {
  return Math.max(...numbers);
}



