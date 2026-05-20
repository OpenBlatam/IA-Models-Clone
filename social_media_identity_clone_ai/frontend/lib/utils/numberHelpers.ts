export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

export const round = (value: number, decimals = 0): number => {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
};

export const formatNumber = (value: number, options?: Intl.NumberFormatOptions): string => {
  return new Intl.NumberFormat('en-US', options).format(value);
};

export const formatCompactNumber = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value);
};

export const random = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export const randomFloat = (min: number, max: number): number => {
  return Math.random() * (max - min) + min;
};

export const isNumeric = (value: unknown): value is number => {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
};

export const parseNumber = (value: string, defaultValue = 0): number => {
  const parsed = Number(value);
  return isNumeric(parsed) ? parsed : defaultValue;
};



