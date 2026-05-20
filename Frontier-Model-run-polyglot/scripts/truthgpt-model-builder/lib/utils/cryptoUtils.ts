/**
 * Utilidades de Criptografía
 * ===========================
 * 
 * Funciones para hashing y encriptación básica
 */

/**
 * Hashea una cadena usando SHA-256
 */
export async function hashSHA256(text: string): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('Web Crypto API not available')
  }

  const encoder = new TextEncoder()
  const data = encoder.encode(text)
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * Hashea una cadena usando SHA-512
 */
export async function hashSHA512(text: string): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('Web Crypto API not available')
  }

  const encoder = new TextEncoder()
  const data = encoder.encode(text)
  const hashBuffer = await crypto.subtle.digest('SHA-512', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * Genera un hash simple (no criptográfico, solo para checksums)
 */
export function simpleHash(text: string): number {
  let hash = 0
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32bit integer
  }
  return Math.abs(hash)
}

/**
 * Genera un token aleatorio seguro
 */
export function generateSecureToken(length: number = 32): string {
  const array = new Uint8Array(length)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(array)
  } else {
    // Fallback para entornos sin crypto
    for (let i = 0; i < length; i++) {
      array[i] = Math.floor(Math.random() * 256)
    }
  }
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('')
}

/**
 * Genera un UUID v4
 */
export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }

  // Fallback
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

/**
 * Codifica texto a Base64
 */
export function encodeBase64(text: string): string {
  if (typeof btoa !== 'undefined') {
    return btoa(text)
  }
  // Fallback para Node.js
  return Buffer.from(text).toString('base64')
}

/**
 * Decodifica Base64 a texto
 */
export function decodeBase64(base64: string): string {
  if (typeof atob !== 'undefined') {
    return atob(base64)
  }
  // Fallback para Node.js
  return Buffer.from(base64, 'base64').toString('utf-8')
}

/**
 * Codifica texto a Base64 URL-safe
 */
export function encodeBase64URL(text: string): string {
  return encodeBase64(text)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '')
}

/**
 * Decodifica Base64 URL-safe
 */
export function decodeBase64URL(base64URL: string): string {
  let base64 = base64URL
    .replace(/-/g, '+')
    .replace(/_/g, '/')

  // Add padding if needed
  while (base64.length % 4) {
    base64 += '='
  }

  return decodeBase64(base64)
}

/**
 * Crea un HMAC (simplificado, para uso no crítico)
 */
export async function createHMAC(
  message: string,
  secret: string,
  algorithm: 'SHA-256' | 'SHA-512' = 'SHA-256'
): Promise<string> {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('Web Crypto API not available')
  }

  const encoder = new TextEncoder()
  const keyData = encoder.encode(secret)
  const messageData = encoder.encode(message)

  const key = await crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'HMAC', hash: algorithm },
    false,
    ['sign']
  )

  const signature = await crypto.subtle.sign('HMAC', key, messageData)
  const hashArray = Array.from(new Uint8Array(signature))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * Verifica un HMAC
 */
export async function verifyHMAC(
  message: string,
  secret: string,
  signature: string,
  algorithm: 'SHA-256' | 'SHA-512' = 'SHA-256'
): Promise<boolean> {
  const expectedSignature = await createHMAC(message, secret, algorithm)
  return expectedSignature === signature
}

/**
 * Genera un nonce
 */
export function generateNonce(length: number = 16): string {
  return generateSecureToken(length)
}

/**
 * Crea un checksum simple
 */
export function createChecksum(data: string): string {
  const hash = simpleHash(data)
  return hash.toString(16).padStart(8, '0')
}

/**
 * Verifica un checksum
 */
export function verifyChecksum(data: string, checksum: string): boolean {
  const calculated = createChecksum(data)
  return calculated === checksum
}






