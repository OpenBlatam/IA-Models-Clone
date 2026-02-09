/**
 * Security utilities
 * 
 * Provides encryption and security functions
 */

/**
 * Hashes a string using Web Crypto API
 */
export async function hashString(data: string, algorithm: "SHA-256" | "SHA-512" = "SHA-256"): Promise<string> {
  if (typeof window === "undefined" || !window.crypto || !window.crypto.subtle) {
    throw new Error("Web Crypto API is not available");
  }

  const encoder = new TextEncoder();
  const dataBuffer = encoder.encode(data);
  const hashBuffer = await window.crypto.subtle.digest(algorithm, dataBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Generates a secure random string
 */
export function generateSecureRandomString(length: number = 32): string {
  if (typeof window === "undefined" || !window.crypto || !window.crypto.getRandomValues) {
    throw new Error("Web Crypto API is not available");
  }

  const array = new Uint8Array(length);
  window.crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * Generates a secure random number
 */
export function generateSecureRandomNumber(min: number = 0, max: number = 1): number {
  if (typeof window === "undefined" || !window.crypto || !window.crypto.getRandomValues) {
    throw new Error("Web Crypto API is not available");
  }

  const range = max - min;
  const randomArray = new Uint32Array(1);
  window.crypto.getRandomValues(randomArray);
  return min + (randomArray[0] / (0xFFFFFFFF + 1)) * range;
}

/**
 * Masks sensitive data
 */
export function maskSensitiveData(data: string, visibleChars: number = 4): string {
  if (data.length <= visibleChars) {
    return "*".repeat(data.length);
  }
  const visible = data.slice(-visibleChars);
  const masked = "*".repeat(data.length - visibleChars);
  return masked + visible;
}

/**
 * Validates password strength
 */
export function validatePasswordStrength(password: string): {
  readonly isValid: boolean;
  readonly score: number;
  readonly feedback: string[];
} {
  const feedback: string[] = [];
  let score = 0;

  if (password.length >= 8) {
    score += 1;
  } else {
    feedback.push("Password must be at least 8 characters long");
  }

  if (/[a-z]/.test(password)) {
    score += 1;
  } else {
    feedback.push("Password must contain lowercase letters");
  }

  if (/[A-Z]/.test(password)) {
    score += 1;
  } else {
    feedback.push("Password must contain uppercase letters");
  }

  if (/[0-9]/.test(password)) {
    score += 1;
  } else {
    feedback.push("Password must contain numbers");
  }

  if (/[^a-zA-Z0-9]/.test(password)) {
    score += 1;
  } else {
    feedback.push("Password should contain special characters");
  }

  return {
    isValid: score >= 4,
    score,
    feedback: score >= 4 ? [] : feedback,
  };
}

/**
 * Sanitizes filename
 */
export function sanitizeFilename(filename: string): string {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, "_")
    .replace(/_{2,}/g, "_")
    .replace(/^_|_$/g, "");
}




