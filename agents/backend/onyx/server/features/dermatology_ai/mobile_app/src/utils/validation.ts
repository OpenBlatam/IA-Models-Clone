/**
 * Validation utilities
 */

export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const validateImageUri = (uri: string): boolean => {
  if (!uri) return false;
  const validProtocols = ['http://', 'https://', 'file://', 'content://'];
  return validProtocols.some(protocol => uri.startsWith(protocol));
};

export const validateVideoUri = (uri: string): boolean => {
  if (!uri) return false;
  const validExtensions = ['.mp4', '.mov', '.avi', '.mkv'];
  const validProtocols = ['http://', 'https://', 'file://', 'content://'];
  return (
    validProtocols.some(protocol => uri.startsWith(protocol)) &&
    validExtensions.some(ext => uri.toLowerCase().includes(ext))
  );
};

export const validateScore = (score: number): boolean => {
  return score >= 0 && score <= 100;
};

export const validateRequired = (value: any): boolean => {
  if (value === null || value === undefined) return false;
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return true;
};

export const validateLength = (value: string, min: number, max: number): boolean => {
  if (!value) return false;
  return value.length >= min && value.length <= max;
};

export const validatePhone = (phone: string): boolean => {
  const phoneRegex = /^\+?[\d\s\-()]+$/;
  return phoneRegex.test(phone) && phone.replace(/\D/g, '').length >= 10;
};

