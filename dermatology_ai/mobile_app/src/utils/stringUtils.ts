/**
 * String utilities
 */

/**
 * Capitalize first letter
 */
export const capitalize = (str: string): string => {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

/**
 * Capitalize each word
 */
export const capitalizeWords = (str: string): string => {
  if (!str) return str;
  return str
    .split(' ')
    .map((word) => capitalize(word))
    .join(' ');
};

/**
 * Truncate string
 */
export const truncate = (str: string, length: number, suffix: string = '...'): string => {
  if (!str || str.length <= length) return str;
  return str.slice(0, length) + suffix;
};

/**
 * Remove accents
 */
export const removeAccents = (str: string): string => {
  return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
};

/**
 * Slugify string
 */
export const slugify = (str: string): string => {
  return removeAccents(str)
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
};

/**
 * Extract initials
 */
export const getInitials = (name: string, maxLength: number = 2): string => {
  if (!name) return '';
  return name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, maxLength);
};

/**
 * Mask string (for sensitive data)
 */
export const maskString = (
  str: string,
  visibleChars: number = 4,
  maskChar: string = '*'
): string => {
  if (!str || str.length <= visibleChars) return str;
  const visible = str.slice(-visibleChars);
  const masked = maskChar.repeat(str.length - visibleChars);
  return masked + visible;
};

/**
 * Format phone number
 */
export const formatPhoneNumber = (phone: string): string => {
  const cleaned = phone.replace(/\D/g, '');
  const match = cleaned.match(/^(\d{3})(\d{3})(\d{4})$/);
  if (match) {
    return `(${match[1]}) ${match[2]}-${match[3]}`;
  }
  return phone;
};

/**
 * Format credit card number
 */
export const formatCreditCard = (card: string): string => {
  const cleaned = card.replace(/\D/g, '');
  const match = cleaned.match(/^(\d{4})(\d{4})(\d{4})(\d{4})$/);
  if (match) {
    return `${match[1]} ${match[2]} ${match[3]} ${match[4]}`;
  }
  return card;
};

/**
 * Generate random string
 */
export const generateRandomString = (length: number = 10): string => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
};

/**
 * Check if string is email
 */
export const isEmail = (str: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(str);
};

/**
 * Check if string is URL
 */
export const isURL = (str: string): boolean => {
  try {
    new URL(str);
    return true;
  } catch {
    return false;
  }
};

