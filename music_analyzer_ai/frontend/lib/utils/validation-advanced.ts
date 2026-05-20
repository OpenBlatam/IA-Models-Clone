/**
 * Advanced validation utility functions.
 * Provides enhanced validation for various data types.
 */

/**
 * Validates an email address.
 * @param email - Email to validate
 * @returns True if valid email
 */
export function isValidEmail(email: string): boolean {
  const emailRegex =
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validates a URL.
 * @param url - URL to validate
 * @returns True if valid URL
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validates a phone number.
 * @param phone - Phone number to validate
 * @param country - Country code (default: 'US')
 * @returns True if valid phone
 */
export function isValidPhone(
  phone: string,
  country: string = 'US'
): boolean {
  const cleaned = phone.replace(/\D/g, '');

  if (country === 'US') {
    return cleaned.length === 10;
  }

  // International: 7-15 digits
  return cleaned.length >= 7 && cleaned.length <= 15;
}

/**
 * Validates a credit card number (Luhn algorithm).
 * @param cardNumber - Card number to validate
 * @returns True if valid card number
 */
export function isValidCardNumber(cardNumber: string): boolean {
  const cleaned = cardNumber.replace(/\D/g, '');

  if (cleaned.length < 13 || cleaned.length > 19) {
    return false;
  }

  // Luhn algorithm
  let sum = 0;
  let isEven = false;

  for (let i = cleaned.length - 1; i >= 0; i--) {
    let digit = parseInt(cleaned[i], 10);

    if (isEven) {
      digit *= 2;
      if (digit > 9) {
        digit -= 9;
      }
    }

    sum += digit;
    isEven = !isEven;
  }

  return sum % 10 === 0;
}

/**
 * Validates a date string.
 * @param date - Date string to validate
 * @returns True if valid date
 */
export function isValidDate(date: string): boolean {
  const dateObj = new Date(date);
  return !isNaN(dateObj.getTime());
}

/**
 * Validates a time string (HH:MM format).
 * @param time - Time string to validate
 * @returns True if valid time
 */
export function isValidTime(time: string): boolean {
  const timeRegex = /^([0-1][0-9]|2[0-3]):[0-5][0-9]$/;
  return timeRegex.test(time);
}

/**
 * Validates a password strength.
 * @param password - Password to validate
 * @param options - Validation options
 * @returns Validation result
 */
export function validatePassword(
  password: string,
  options: {
    minLength?: number;
    requireUppercase?: boolean;
    requireLowercase?: boolean;
    requireNumbers?: boolean;
    requireSpecial?: boolean;
  } = {}
): {
  valid: boolean;
  errors: string[];
} {
  const {
    minLength = 8,
    requireUppercase = true,
    requireLowercase = true,
    requireNumbers = true,
    requireSpecial = false,
  } = options;

  const errors: string[] = [];

  if (password.length < minLength) {
    errors.push(`Password must be at least ${minLength} characters`);
  }

  if (requireUppercase && !/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }

  if (requireLowercase && !/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }

  if (requireNumbers && !/\d/.test(password)) {
    errors.push('Password must contain at least one number');
  }

  if (requireSpecial && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    errors.push('Password must contain at least one special character');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validates a credit card expiry date.
 * @param expiry - Expiry date (MM/YY format)
 * @returns True if valid and not expired
 */
export function isValidCardExpiry(expiry: string): boolean {
  const expiryRegex = /^(0[1-9]|1[0-2])\/([0-9]{2})$/;
  if (!expiryRegex.test(expiry)) {
    return false;
  }

  const [month, year] = expiry.split('/');
  const expiryDate = new Date(2000 + parseInt(year, 10), parseInt(month, 10) - 1);
  const now = new Date();

  return expiryDate > now;
}

