export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

export const lerp = (start: number, end: number, t: number): number => {
  return start + (end - start) * t;
};

export const mapRange = (
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number => {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
};

export const round = (value: number, decimals: number = 0): number => {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
};

export const floor = (value: number, decimals: number = 0): number => {
  const factor = Math.pow(10, decimals);
  return Math.floor(value * factor) / factor;
};

export const ceil = (value: number, decimals: number = 0): number => {
  const factor = Math.pow(10, decimals);
  return Math.ceil(value * factor) / factor;
};

export const random = (min: number = 0, max: number = 1): number => {
  return Math.random() * (max - min) + min;
};

export const randomInt = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const randomItem = <T>(array: readonly T[]): T | undefined => {
  if (array.length === 0) return undefined;
  return array[randomInt(0, array.length - 1)];
};

export const randomItems = <T>(array: readonly T[], count: number): T[] => {
  const shuffled = [...array].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(count, array.length));
};

export const percentage = (value: number, total: number): number => {
  if (total === 0) return 0;
  return (value / total) * 100;
};

export const percentageOf = (percent: number, total: number): number => {
  return (percent / 100) * total;
};

export const average = (numbers: readonly number[]): number => {
  if (numbers.length === 0) return 0;
  return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;
};

export const sum = (numbers: readonly number[]): number => {
  return numbers.reduce((sum, num) => sum + num, 0);
};

export const min = (numbers: readonly number[]): number => {
  return Math.min(...numbers);
};

export const max = (numbers: readonly number[]): number => {
  return Math.max(...numbers);
};

export const median = (numbers: readonly number[]): number => {
  if (numbers.length === 0) return 0;
  const sorted = [...numbers].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
};

export const distance = (
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number => {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
};

export const isEven = (value: number): boolean => {
  return value % 2 === 0;
};

export const isOdd = (value: number): boolean => {
  return value % 2 !== 0;
};

export const isBetween = (
  value: number,
  min: number,
  max: number,
  inclusive: boolean = true
): boolean => {
  return inclusive ? value >= min && value <= max : value > min && value < max;
};





