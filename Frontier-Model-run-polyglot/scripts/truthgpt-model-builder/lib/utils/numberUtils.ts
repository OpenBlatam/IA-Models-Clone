/**
 * Utilidades para Números
 * ======================
 * 
 * Funciones útiles para trabajar con números
 */

// ============================================================================
// VALIDACIÓN
// ============================================================================

/**
 * Verifica si un valor es un número válido
 */
export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value)
}

/**
 * Verifica si un número es par
 */
export function isEven(num: number): boolean {
  return num % 2 === 0
}

/**
 * Verifica si un número es impar
 */
export function isOdd(num: number): boolean {
  return num % 2 !== 0
}

/**
 * Verifica si un número es entero
 */
export function isInteger(num: number): boolean {
  return Number.isInteger(num)
}

/**
 * Verifica si un número es positivo
 */
export function isPositive(num: number): boolean {
  return num > 0
}

/**
 * Verifica si un número es negativo
 */
export function isNegative(num: number): boolean {
  return num < 0
}

/**
 * Verifica si un número está en un rango
 */
export function isInRange(num: number, min: number, max: number, inclusive: boolean = true): boolean {
  return inclusive ? num >= min && num <= max : num > min && num < max
}

// ============================================================================
// TRANSFORMACIONES
// ============================================================================

/**
 * Convierte un valor a número de forma segura
 */
export function toNumber(value: unknown, defaultValue: number = 0): number {
  if (isNumber(value)) return value
  const parsed = Number(value)
  return isNumber(parsed) ? parsed : defaultValue
}

/**
 * Redondea un número a N decimales
 */
export function round(num: number, decimals: number = 0): number {
  const factor = Math.pow(10, decimals)
  return Math.round(num * factor) / factor
}

/**
 * Trunca un número a N decimales
 */
export function truncate(num: number, decimals: number = 0): number {
  const factor = Math.pow(10, decimals)
  return Math.trunc(num * factor) / factor
}

/**
 * Limita un número entre un mínimo y máximo
 */
export function clamp(num: number, min: number, max: number): number {
  return Math.min(Math.max(num, min), max)
}

/**
 * Normaliza un número a un rango 0-1
 */
export function normalize(num: number, min: number, max: number): number {
  return (num - min) / (max - min)
}

/**
 * Desnormaliza un número de 0-1 a un rango
 */
export function denormalize(num: number, min: number, max: number): number {
  return num * (max - min) + min
}

/**
 * Mapea un número de un rango a otro
 */
export function mapRange(
  num: number,
  fromMin: number,
  fromMax: number,
  toMin: number,
  toMax: number
): number {
  return denormalize(normalize(num, fromMin, fromMax), toMin, toMax)
}

// ============================================================================
// OPERACIONES
// ============================================================================

/**
 * Calcula el porcentaje de un número
 */
export function percentage(num: number, total: number): number {
  return total === 0 ? 0 : (num / total) * 100
}

/**
 * Calcula el valor de un porcentaje
 */
export function percentOf(percent: number, total: number): number {
  return (percent / 100) * total
}

/**
 * Calcula el incremento porcentual
 */
export function percentChange(oldValue: number, newValue: number): number {
  if (oldValue === 0) return newValue === 0 ? 0 : 100
  return ((newValue - oldValue) / oldValue) * 100
}

/**
 * Suma un array de números
 */
export function sum(numbers: number[]): number {
  return numbers.reduce((acc, num) => acc + num, 0)
}

/**
 * Calcula el promedio de un array de números
 */
export function average(numbers: number[]): number {
  if (numbers.length === 0) return 0
  return sum(numbers) / numbers.length
}

/**
 * Calcula la mediana de un array de números
 */
export function median(numbers: number[]): number {
  if (numbers.length === 0) return 0
  
  const sorted = [...numbers].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid]
}

/**
 * Calcula la moda de un array de números
 */
export function mode(numbers: number[]): number | null {
  if (numbers.length === 0) return null
  
  const frequency: Record<number, number> = {}
  let maxFreq = 0
  let mode: number | null = null
  
  for (const num of numbers) {
    frequency[num] = (frequency[num] || 0) + 1
    if (frequency[num] > maxFreq) {
      maxFreq = frequency[num]
      mode = num
    }
  }
  
  return mode
}

/**
 * Calcula la desviación estándar
 */
export function standardDeviation(numbers: number[]): number {
  if (numbers.length === 0) return 0
  
  const avg = average(numbers)
  const squareDiffs = numbers.map(num => Math.pow(num - avg, 2))
  const avgSquareDiff = average(squareDiffs)
  
  return Math.sqrt(avgSquareDiff)
}

/**
 * Calcula la varianza
 */
export function variance(numbers: number[]): number {
  if (numbers.length === 0) return 0
  
  const avg = average(numbers)
  const squareDiffs = numbers.map(num => Math.pow(num - avg, 2))
  
  return average(squareDiffs)
}

// ============================================================================
// UTILIDADES
// ============================================================================

/**
 * Genera un número aleatorio entre min y max
 */
export function random(min: number = 0, max: number = 1): number {
  return Math.random() * (max - min) + min
}

/**
 * Genera un entero aleatorio entre min y max (inclusive)
 */
export function randomInt(min: number, max: number): number {
  return Math.floor(random(min, max + 1))
}

/**
 * Formatea un número con separadores de miles
 */
export function formatNumber(num: number, decimals: number = 0, locale: string = 'es-ES'): string {
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(num)
}

/**
 * Formatea un número como porcentaje
 */
export function formatPercent(num: number, decimals: number = 1): string {
  return `${round(num * 100, decimals)}%`
}

/**
 * Formatea bytes a formato legible
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return `${round(bytes / Math.pow(k, i), dm)} ${sizes[i]}`
}

/**
 * Convierte un número a formato compacto (1K, 1M, etc.)
 */
export function formatCompact(num: number, decimals: number = 1): string {
  if (num < 1000) return String(num)
  if (num < 1000000) return `${round(num / 1000, decimals)}K`
  if (num < 1000000000) return `${round(num / 1000000, decimals)}M`
  if (num < 1000000000000) return `${round(num / 1000000000, decimals)}B`
  return `${round(num / 1000000000000, decimals)}T`
}

/**
 * Convierte un string a número de forma segura
 */
export function parseNumber(value: string, defaultValue: number = 0): number {
  const parsed = parseFloat(value)
  return isNumber(parsed) ? parsed : defaultValue
}

/**
 * Verifica si un número es primo
 */
export function isPrime(num: number): boolean {
  if (num < 2) return false
  if (num === 2) return true
  if (num % 2 === 0) return false
  
  for (let i = 3; i <= Math.sqrt(num); i += 2) {
    if (num % i === 0) return false
  }
  
  return true
}

/**
 * Calcula el factorial de un número
 */
export function factorial(n: number): number {
  if (n < 0) return NaN
  if (n === 0 || n === 1) return 1
  
  let result = 1
  for (let i = 2; i <= n; i++) {
    result *= i
  }
  
  return result
}







