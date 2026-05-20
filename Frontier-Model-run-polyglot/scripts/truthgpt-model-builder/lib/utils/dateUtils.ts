/**
 * Utilidades para Fechas
 * =====================
 * 
 * Funciones útiles para trabajar con fechas
 */

// ============================================================================
// CREACIÓN Y CONVERSIÓN
// ============================================================================

/**
 * Crea una fecha desde diferentes formatos
 */
export function createDate(input: Date | string | number): Date {
  if (input instanceof Date) return input
  if (typeof input === 'number') return new Date(input)
  return new Date(input)
}

/**
 * Convierte una fecha a timestamp
 */
export function toTimestamp(date: Date | string): number {
  return createDate(date).getTime()
}

/**
 * Convierte un timestamp a fecha
 */
export function fromTimestamp(timestamp: number): Date {
  return new Date(timestamp)
}

// ============================================================================
// FORMATO
// ============================================================================

/**
 * Formatea una fecha en formato ISO
 */
export function toISO(date: Date | string): string {
  return createDate(date).toISOString()
}

/**
 * Formatea una fecha en formato local
 */
export function toLocalString(date: Date | string, locale: string = 'es-ES'): string {
  return createDate(date).toLocaleString(locale)
}

/**
 * Formatea solo la fecha (sin hora)
 */
export function formatDateOnly(date: Date | string, locale: string = 'es-ES'): string {
  return createDate(date).toLocaleDateString(locale)
}

/**
 * Formatea solo la hora (sin fecha)
 */
export function formatTimeOnly(date: Date | string, locale: string = 'es-ES'): string {
  return createDate(date).toLocaleTimeString(locale)
}

// ============================================================================
// OPERACIONES
// ============================================================================

/**
 * Agrega días a una fecha
 */
export function addDays(date: Date | string, days: number): Date {
  const result = new Date(createDate(date))
  result.setDate(result.getDate() + days)
  return result
}

/**
 * Agrega horas a una fecha
 */
export function addHours(date: Date | string, hours: number): Date {
  const result = new Date(createDate(date))
  result.setHours(result.getHours() + hours)
  return result
}

/**
 * Agrega minutos a una fecha
 */
export function addMinutes(date: Date | string, minutes: number): Date {
  const result = new Date(createDate(date))
  result.setMinutes(result.getMinutes() + minutes)
  return result
}

/**
 * Agrega meses a una fecha
 */
export function addMonths(date: Date | string, months: number): Date {
  const result = new Date(createDate(date))
  result.setMonth(result.getMonth() + months)
  return result
}

/**
 * Agrega años a una fecha
 */
export function addYears(date: Date | string, years: number): Date {
  const result = new Date(createDate(date))
  result.setFullYear(result.getFullYear() + years)
  return result
}

/**
 * Resta días de una fecha
 */
export function subtractDays(date: Date | string, days: number): Date {
  return addDays(date, -days)
}

/**
 * Resta horas de una fecha
 */
export function subtractHours(date: Date | string, hours: number): Date {
  return addHours(date, -hours)
}

// ============================================================================
// COMPARACIÓN
// ============================================================================

/**
 * Verifica si una fecha es anterior a otra
 */
export function isBefore(date1: Date | string, date2: Date | string): boolean {
  return createDate(date1).getTime() < createDate(date2).getTime()
}

/**
 * Verifica si una fecha es posterior a otra
 */
export function isAfter(date1: Date | string, date2: Date | string): boolean {
  return createDate(date1).getTime() > createDate(date2).getTime()
}

/**
 * Verifica si una fecha está entre dos fechas
 */
export function isBetween(date: Date | string, start: Date | string, end: Date | string): boolean {
  const d = createDate(date).getTime()
  const s = createDate(start).getTime()
  const e = createDate(end).getTime()
  return d >= s && d <= e
}

/**
 * Verifica si dos fechas son del mismo día
 */
export function isSameDay(date1: Date | string, date2: Date | string): boolean {
  const d1 = createDate(date1)
  const d2 = createDate(date2)
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
  )
}

/**
 * Verifica si una fecha es hoy
 */
export function isToday(date: Date | string): boolean {
  return isSameDay(date, new Date())
}

/**
 * Verifica si una fecha es ayer
 */
export function isYesterday(date: Date | string): boolean {
  return isSameDay(date, subtractDays(new Date(), 1))
}

/**
 * Verifica si una fecha es mañana
 */
export function isTomorrow(date: Date | string): boolean {
  return isSameDay(date, addDays(new Date(), 1))
}

// ============================================================================
// DIFERENCIAS
// ============================================================================

/**
 * Calcula la diferencia en días entre dos fechas
 */
export function diffInDays(date1: Date | string, date2: Date | string): number {
  const d1 = createDate(date1).getTime()
  const d2 = createDate(date2).getTime()
  return Math.floor((d2 - d1) / (1000 * 60 * 60 * 24))
}

/**
 * Calcula la diferencia en horas entre dos fechas
 */
export function diffInHours(date1: Date | string, date2: Date | string): number {
  const d1 = createDate(date1).getTime()
  const d2 = createDate(date2).getTime()
  return Math.floor((d2 - d1) / (1000 * 60 * 60))
}

/**
 * Calcula la diferencia en minutos entre dos fechas
 */
export function diffInMinutes(date1: Date | string, date2: Date | string): number {
  const d1 = createDate(date1).getTime()
  const d2 = createDate(date2).getTime()
  return Math.floor((d2 - d1) / (1000 * 60))
}

/**
 * Calcula la diferencia en segundos entre dos fechas
 */
export function diffInSeconds(date1: Date | string, date2: Date | string): number {
  const d1 = createDate(date1).getTime()
  const d2 = createDate(date2).getTime()
  return Math.floor((d2 - d1) / 1000)
}

// ============================================================================
// UTILIDADES
// ============================================================================

/**
 * Obtiene el inicio del día
 */
export function startOfDay(date: Date | string): Date {
  const result = new Date(createDate(date))
  result.setHours(0, 0, 0, 0)
  return result
}

/**
 * Obtiene el final del día
 */
export function endOfDay(date: Date | string): Date {
  const result = new Date(createDate(date))
  result.setHours(23, 59, 59, 999)
  return result
}

/**
 * Obtiene el inicio de la semana
 */
export function startOfWeek(date: Date | string, weekStartsOn: number = 0): Date {
  const result = new Date(createDate(date))
  const day = result.getDay()
  const diff = (day < weekStartsOn ? 7 : 0) + day - weekStartsOn
  result.setDate(result.getDate() - diff)
  return startOfDay(result)
}

/**
 * Obtiene el final de la semana
 */
export function endOfWeek(date: Date | string, weekStartsOn: number = 0): Date {
  return addDays(startOfWeek(date, weekStartsOn), 6)
}

/**
 * Obtiene el inicio del mes
 */
export function startOfMonth(date: Date | string): Date {
  const result = new Date(createDate(date))
  result.setDate(1)
  return startOfDay(result)
}

/**
 * Obtiene el final del mes
 */
export function endOfMonth(date: Date | string): Date {
  const result = new Date(createDate(date))
  result.setMonth(result.getMonth() + 1, 0)
  return endOfDay(result)
}

/**
 * Obtiene el inicio del año
 */
export function startOfYear(date: Date | string): Date {
  const result = new Date(createDate(date))
  result.setMonth(0, 1)
  return startOfDay(result)
}

/**
 * Obtiene el final del año
 */
export function endOfYear(date: Date | string): Date {
  const result = new Date(createDate(date))
  result.setMonth(11, 31)
  return endOfDay(result)
}

/**
 * Obtiene la edad desde una fecha de nacimiento
 */
export function getAge(birthDate: Date | string): number {
  const today = new Date()
  const birth = createDate(birthDate)
  let age = today.getFullYear() - birth.getFullYear()
  const monthDiff = today.getMonth() - birth.getMonth()
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--
  }
  
  return age
}

/**
 * Formatea tiempo relativo (hace X tiempo)
 */
export function formatRelative(date: Date | string): string {
  const now = new Date()
  const then = createDate(date)
  const diffInSeconds = Math.floor((now.getTime() - then.getTime()) / 1000)

  if (diffInSeconds < 60) return 'hace unos segundos'
  if (diffInSeconds < 3600) return `hace ${Math.floor(diffInSeconds / 60)} minutos`
  if (diffInSeconds < 86400) return `hace ${Math.floor(diffInSeconds / 3600)} horas`
  if (diffInSeconds < 604800) return `hace ${Math.floor(diffInSeconds / 86400)} días`
  if (diffInSeconds < 2592000) return `hace ${Math.floor(diffInSeconds / 604800)} semanas`
  if (diffInSeconds < 31536000) return `hace ${Math.floor(diffInSeconds / 2592000)} meses`
  return `hace ${Math.floor(diffInSeconds / 31536000)} años`
}







