/**
 * Auth helper utilities
 */

import { isValidEmail } from '@/utils/strings';
import { LIMITS } from '@/utils/constants';
import type { LoginFormData, RegisterFormData } from './auth-types';

export function validateLoginForm(data: LoginFormData): {
  valid: boolean;
  errors: Partial<Record<keyof LoginFormData, string>>;
} {
  const errors: Partial<Record<keyof LoginFormData, string>> = {};

  if (!data.email) {
    errors.email = 'Email is required';
  } else if (!isValidEmail(data.email)) {
    errors.email = 'Invalid email address';
  }

  if (!data.password) {
    errors.password = 'Password is required';
  } else if (data.password.length < LIMITS.password_min_length) {
    errors.password = `Password must be at least ${LIMITS.password_min_length} characters`;
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
}

export function validateRegisterForm(data: RegisterFormData): {
  valid: boolean;
  errors: Partial<Record<keyof RegisterFormData, string>>;
} {
  const errors: Partial<Record<keyof RegisterFormData, string>> = {};

  if (!data.email) {
    errors.email = 'Email is required';
  } else if (!isValidEmail(data.email)) {
    errors.email = 'Invalid email address';
  }

  if (!data.password) {
    errors.password = 'Password is required';
  } else if (data.password.length < LIMITS.password_min_length) {
    errors.password = `Password must be at least ${LIMITS.password_min_length} characters`;
  }

  if (!data.confirmPassword) {
    errors.confirmPassword = 'Please confirm your password';
  } else if (data.password !== data.confirmPassword) {
    errors.confirmPassword = 'Passwords do not match';
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
}

export function maskEmail(email: string): string {
  const [localPart, domain] = email.split('@');
  if (!domain) return email;

  const maskedLocal =
    localPart.length > 2
      ? localPart.substring(0, 2) + '*'.repeat(localPart.length - 2)
      : '*'.repeat(localPart.length);

  return `${maskedLocal}@${domain}`;
}

