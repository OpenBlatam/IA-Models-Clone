import { VALIDATION } from './constants';

export const validateUsername = (username: string): string | null => {
  const trimmed = username.trim();
  
  if (!trimmed) {
    return 'Username is required';
  }
  
  if (trimmed.length < VALIDATION.MIN_USERNAME_LENGTH) {
    return `Username must be at least ${VALIDATION.MIN_USERNAME_LENGTH} character(s)`;
  }
  
  if (trimmed.length > VALIDATION.MAX_USERNAME_LENGTH) {
    return `Username must be at most ${VALIDATION.MAX_USERNAME_LENGTH} characters`;
  }
  
  return null;
};

export const validateRequired = (value: string, fieldName: string): string | null => {
  if (!value.trim()) {
    return `${fieldName} is required`;
  }
  return null;
};

export const validateEmail = (email: string): string | null => {
  if (!email.trim()) {
    return 'Email is required';
  }
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return 'Invalid email format';
  }
  
  return null;
};

export const validateMinLength = (value: string, minLength: number, fieldName: string): string | null => {
  if (value.length < minLength) {
    return `${fieldName} must be at least ${minLength} character(s)`;
  }
  return null;
};

export const validateMaxLength = (value: string, maxLength: number, fieldName: string): string | null => {
  if (value.length > maxLength) {
    return `${fieldName} must be at most ${maxLength} characters`;
  }
  return null;
};



