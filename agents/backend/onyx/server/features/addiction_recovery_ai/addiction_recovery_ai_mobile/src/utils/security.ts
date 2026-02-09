import { sanitizeString } from './sanitize';

export function validateAndSanitizeInput(
  input: string,
  maxLength?: number
): { isValid: boolean; sanitized: string; error?: string } {
  if (!input || typeof input !== 'string') {
    return {
      isValid: false,
      sanitized: '',
      error: 'Input must be a non-empty string',
    };
  }

  if (maxLength && input.length > maxLength) {
    return {
      isValid: false,
      sanitized: '',
      error: `Input must be less than ${maxLength} characters`,
    };
  }

  const sanitized = sanitizeString(input);

  if (sanitized.length === 0) {
    return {
      isValid: false,
      sanitized: '',
      error: 'Input contains only invalid characters',
    };
  }

  return {
    isValid: true,
    sanitized,
  };
}

export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export function validateURL(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export function maskSensitiveData(data: string, visibleChars = 4): string {
  if (data.length <= visibleChars) {
    return '*'.repeat(data.length);
  }

  const visible = data.slice(-visibleChars);
  const masked = '*'.repeat(data.length - visibleChars);
  return masked + visible;
}

export function generateSecureToken(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let token = '';
  
  for (let i = 0; i < length; i++) {
    token += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  
  return token;
}

