/**
 * Advanced validation utilities
 */

import { z } from 'zod';

// Email validation
export const emailSchema = z.string().email('Email inválido');

// URL validation
export const urlSchema = z.string().url('URL inválida');

// Phone validation
export const phoneSchema = z.string().regex(/^\+?[\d\s-()]+$/, 'Teléfono inválido');

// Password validation
export const passwordSchema = z
  .string()
  .min(8, 'La contraseña debe tener al menos 8 caracteres')
  .regex(/[A-Z]/, 'La contraseña debe contener al menos una mayúscula')
  .regex(/[a-z]/, 'La contraseña debe contener al menos una minúscula')
  .regex(/[0-9]/, 'La contraseña debe contener al menos un número');

// Credit card validation (Luhn algorithm)
export function validateCreditCard(cardNumber: string): boolean {
  const cleaned = cardNumber.replace(/\s/g, '');
  if (!/^\d+$/.test(cleaned) || cleaned.length < 13) {
    return false;
  }

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

// Validate date range
export function validateDateRange(start: Date, end: Date): boolean {
  return start <= end;
}

// Validate age
export function validateAge(birthDate: Date, minAge: number = 18): boolean {
  const today = new Date();
  const age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
    return age - 1 >= minAge;
  }
  
  return age >= minAge;
}

// Validate file size
export function validateFileSize(file: File, maxSizeMB: number): boolean {
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  return file.size <= maxSizeBytes;
}

// Validate file type
export function validateFileType(file: File, allowedTypes: string[]): boolean {
  return allowedTypes.some((type) => {
    if (type.endsWith('/*')) {
      return file.type.startsWith(type.slice(0, -1));
    }
    return file.type === type;
  });
}

// Validate image dimensions
export function validateImageDimensions(
  file: File,
  maxWidth: number,
  maxHeight: number
): Promise<boolean> {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img.width <= maxWidth && img.height <= maxHeight);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(false);
    };

    img.src = url;
  });
}

// Validate IBAN (simplified)
export function validateIBAN(iban: string): boolean {
  const cleaned = iban.replace(/\s/g, '').toUpperCase();
  
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]+$/.test(cleaned)) {
    return false;
  }

  // Move first 4 characters to end
  const rearranged = cleaned.slice(4) + cleaned.slice(0, 4);
  
  // Convert letters to numbers (A=10, B=11, etc.)
  const numeric = rearranged.replace(/[A-Z]/g, (char) => {
    return (char.charCodeAt(0) - 55).toString();
  });

  // Calculate mod 97
  let remainder = '';
  for (let i = 0; i < numeric.length; i += 7) {
    remainder = (parseInt(remainder + numeric.slice(i, i + 7), 10) % 97).toString();
  }

  return parseInt(remainder, 10) === 1;
}



