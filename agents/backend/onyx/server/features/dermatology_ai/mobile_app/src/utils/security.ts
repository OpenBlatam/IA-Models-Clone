import * as Crypto from 'expo-crypto';

/**
 * Hash a string (for passwords, tokens, etc.)
 */
export const hashString = async (input: string): Promise<string> => {
  try {
    const hash = await Crypto.digestStringAsync(
      Crypto.CryptoDigestAlgorithm.SHA256,
      input
    );
    return hash;
  } catch (error) {
    console.error('Error hashing string:', error);
    throw error;
  }
};

/**
 * Generate a random token
 */
export const generateToken = async (length: number = 32): Promise<string> => {
  try {
    const randomBytes = await Crypto.getRandomBytesAsync(length);
    return Array.from(randomBytes)
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('');
  } catch (error) {
    console.error('Error generating token:', error);
    throw error;
  }
};

/**
 * Encrypt sensitive data (basic implementation)
 */
export const encryptData = async (data: string, key: string): Promise<string> => {
  // Note: This is a basic implementation
  // For production, use a proper encryption library
  const hash = await hashString(key);
  return hash; // Simplified
};

/**
 * Validate token format
 */
export const isValidToken = (token: string): boolean => {
  return token.length >= 32 && /^[a-f0-9]+$/i.test(token);
};

