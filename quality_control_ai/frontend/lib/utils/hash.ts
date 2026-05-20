export const hash = async (str: string, algorithm: 'SHA-1' | 'SHA-256' | 'SHA-512' = 'SHA-256'): Promise<string> => {
  if (typeof window === 'undefined' || !window.crypto || !window.crypto.subtle) {
    throw new Error('Web Crypto API not available');
  }

  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await window.crypto.subtle.digest(algorithm, data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
};

export const simpleHash = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash;
};

export const hashObject = async (obj: unknown, algorithm: 'SHA-1' | 'SHA-256' | 'SHA-512' = 'SHA-256'): Promise<string> => {
  const str = JSON.stringify(obj);
  return hash(str, algorithm);
};

export const simpleHashObject = (obj: unknown): number => {
  const str = JSON.stringify(obj);
  return simpleHash(str);
};

