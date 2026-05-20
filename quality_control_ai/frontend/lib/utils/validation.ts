export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

export const isValidNumber = (value: string | number): boolean => {
  return !isNaN(Number(value)) && isFinite(Number(value));
};

export const isPositiveNumber = (value: string | number): boolean => {
  return isValidNumber(value) && Number(value) > 0;
};

export const isNegativeNumber = (value: string | number): boolean => {
  return isValidNumber(value) && Number(value) < 0;
};

export const isInteger = (value: string | number): boolean => {
  return Number.isInteger(Number(value));
};

export const isFloat = (value: string | number): boolean => {
  return isValidNumber(value) && !isInteger(value);
};

export const isInRange = (
  value: number,
  min: number,
  max: number
): boolean => {
  return value >= min && value <= max;
};

export const isAlpha = (str: string): boolean => {
  return /^[a-zA-Z]+$/.test(str);
};

export const isAlphaNumeric = (str: string): boolean => {
  return /^[a-zA-Z0-9]+$/.test(str);
};

export const isNumeric = (str: string): boolean => {
  return /^[0-9]+$/.test(str);
};

export const isEmpty = (value: unknown): boolean => {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') return value.trim().length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
};

export const isNotEmpty = (value: unknown): boolean => {
  return !isEmpty(value);
};

export const hasMinLength = (str: string, min: number): boolean => {
  return str.length >= min;
};

export const hasMaxLength = (str: string, max: number): boolean => {
  return str.length <= max;
};

export const hasLength = (str: string, length: number): boolean => {
  return str.length === length;
};

export const matches = (str: string, pattern: RegExp | string): boolean => {
  const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
  return regex.test(str);
};

export const isPhoneNumber = (phone: string): boolean => {
  const phoneRegex = /^[\d\s\-\+\(\)]+$/;
  return phoneRegex.test(phone) && phone.replace(/\D/g, '').length >= 10;
};

export const isCreditCard = (card: string): boolean => {
  const cardRegex = /^[\d\s-]+$/;
  const digits = card.replace(/\D/g, '');
  return cardRegex.test(card) && digits.length >= 13 && digits.length <= 19;
};

export const isDate = (date: string | Date): boolean => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return !isNaN(dateObj.getTime());
};

export const isFutureDate = (date: string | Date): boolean => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return isDate(dateObj) && dateObj.getTime() > Date.now();
};

export const isPastDate = (date: string | Date): boolean => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return isDate(dateObj) && dateObj.getTime() < Date.now();
};

