/**
 * Utilidades de Internacionalización
 * ==================================
 * 
 * Funciones para manejar múltiples idiomas
 */

// ============================================================================
// TIPOS
// ============================================================================

export type Locale = 'es' | 'en'
export type TranslationKey = string
export type Translations = Record<TranslationKey, string>

// ============================================================================
// TRADUCCIONES
// ============================================================================

const translations: Record<Locale, Translations> = {
  es: {
    // General
    'common.save': 'Guardar',
    'common.cancel': 'Cancelar',
    'common.delete': 'Eliminar',
    'common.edit': 'Editar',
    'common.create': 'Crear',
    'common.close': 'Cerrar',
    'common.loading': 'Cargando...',
    'common.error': 'Error',
    'common.success': 'Éxito',
    'common.warning': 'Advertencia',
    'common.info': 'Información',
    
    // Model Status
    'model.status.idle': 'Inactivo',
    'model.status.validating': 'Validando',
    'model.status.creating': 'Creando',
    'model.status.compiling': 'Compilando',
    'model.status.training': 'Entrenando',
    'model.status.evaluating': 'Evaluando',
    'model.status.predicting': 'Prediciendo',
    'model.status.completed': 'Completado',
    'model.status.failed': 'Fallido',
    'model.status.cancelled': 'Cancelado',
    'model.status.paused': 'Pausado',
    
    // Model Types
    'model.type.classification': 'Clasificación',
    'model.type.regression': 'Regresión',
    'model.type.nlp': 'Procesamiento de Lenguaje Natural',
    'model.type.vision': 'Visión',
    'model.type.time-series': 'Series de Tiempo',
    'model.type.generative': 'Generativo',
    'model.type.custom': 'Personalizado',
    
    // Errors
    'error.network': 'Error de conexión',
    'error.timeout': 'Tiempo de espera agotado',
    'error.validation': 'Error de validación',
    'error.api': 'Error de API',
    'error.unknown': 'Error desconocido',
    'error.rate-limit': 'Límite de tasa excedido',
    
    // Validation
    'validation.required': 'Este campo es requerido',
    'validation.email': 'Email inválido',
    'validation.url': 'URL inválida',
    'validation.min-length': 'Debe tener al menos {min} caracteres',
    'validation.max-length': 'Debe tener máximo {max} caracteres',
    'validation.min': 'Debe ser mayor o igual a {min}',
    'validation.max': 'Debe ser menor o igual a {max}',
    
    // Messages
    'message.model.created': 'Modelo creado exitosamente',
    'message.model.updated': 'Modelo actualizado exitosamente',
    'message.model.deleted': 'Modelo eliminado exitosamente',
    'message.model.failed': 'Error al crear el modelo',
    
    // Actions
    'action.create-model': 'Crear Modelo',
    'action.edit-model': 'Editar Modelo',
    'action.delete-model': 'Eliminar Modelo',
    'action.save-model': 'Guardar Modelo',
    'action.cancel': 'Cancelar'
  },
  en: {
    // General
    'common.save': 'Save',
    'common.cancel': 'Cancel',
    'common.delete': 'Delete',
    'common.edit': 'Edit',
    'common.create': 'Create',
    'common.close': 'Close',
    'common.loading': 'Loading...',
    'common.error': 'Error',
    'common.success': 'Success',
    'common.warning': 'Warning',
    'common.info': 'Information',
    
    // Model Status
    'model.status.idle': 'Idle',
    'model.status.validating': 'Validating',
    'model.status.creating': 'Creating',
    'model.status.compiling': 'Compiling',
    'model.status.training': 'Training',
    'model.status.evaluating': 'Evaluating',
    'model.status.predicting': 'Predicting',
    'model.status.completed': 'Completed',
    'model.status.failed': 'Failed',
    'model.status.cancelled': 'Cancelled',
    'model.status.paused': 'Paused',
    
    // Model Types
    'model.type.classification': 'Classification',
    'model.type.regression': 'Regression',
    'model.type.nlp': 'Natural Language Processing',
    'model.type.vision': 'Vision',
    'model.type.time-series': 'Time Series',
    'model.type.generative': 'Generative',
    'model.type.custom': 'Custom',
    
    // Errors
    'error.network': 'Network error',
    'error.timeout': 'Timeout error',
    'error.validation': 'Validation error',
    'error.api': 'API error',
    'error.unknown': 'Unknown error',
    'error.rate-limit': 'Rate limit exceeded',
    
    // Validation
    'validation.required': 'This field is required',
    'validation.email': 'Invalid email',
    'validation.url': 'Invalid URL',
    'validation.min-length': 'Must have at least {min} characters',
    'validation.max-length': 'Must have at most {max} characters',
    'validation.min': 'Must be greater than or equal to {min}',
    'validation.max': 'Must be less than or equal to {max}',
    
    // Messages
    'message.model.created': 'Model created successfully',
    'message.model.updated': 'Model updated successfully',
    'message.model.deleted': 'Model deleted successfully',
    'message.model.failed': 'Error creating model',
    
    // Actions
    'action.create-model': 'Create Model',
    'action.edit-model': 'Edit Model',
    'action.delete-model': 'Delete Model',
    'action.save-model': 'Save Model',
    'action.cancel': 'Cancel'
  }
}

// ============================================================================
// FUNCIONES
// ============================================================================

let currentLocale: Locale = 'es'

/**
 * Establece el locale actual
 */
export function setLocale(locale: Locale): void {
  currentLocale = locale
}

/**
 * Obtiene el locale actual
 */
export function getLocale(): Locale {
  return currentLocale
}

/**
 * Traduce una clave
 */
export function t(key: TranslationKey, params?: Record<string, string | number>): string {
  const translation = translations[currentLocale][key] || translations['es'][key] || key
  
  if (!params) return translation
  
  return Object.entries(params).reduce(
    (str, [param, value]) => str.replace(`{${param}}`, String(value)),
    translation
  )
}

/**
 * Verifica si existe una traducción
 */
export function hasTranslation(key: TranslationKey): boolean {
  return key in translations[currentLocale] || key in translations['es']
}

/**
 * Obtiene todas las traducciones del locale actual
 */
export function getAllTranslations(): Translations {
  return translations[currentLocale]
}

/**
 * Agrega traducciones personalizadas
 */
export function addTranslations(locale: Locale, newTranslations: Partial<Translations>): void {
  translations[locale] = {
    ...translations[locale],
    ...newTranslations
  }
}

/**
 * Formatea un número según el locale
 */
export function formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
  return new Intl.NumberFormat(currentLocale === 'es' ? 'es-ES' : 'en-US', options).format(value)
}

/**
 * Formatea una fecha según el locale
 */
export function formatDate(date: Date | string, options?: Intl.DateTimeFormatOptions): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  return new Intl.DateTimeFormat(
    currentLocale === 'es' ? 'es-ES' : 'en-US',
    options
  ).format(dateObj)
}

/**
 * Formatea una fecha relativa
 */
export function formatRelativeTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - dateObj.getTime()) / 1000)

  if (currentLocale === 'es') {
    if (diffInSeconds < 60) return 'hace unos segundos'
    if (diffInSeconds < 3600) return `hace ${Math.floor(diffInSeconds / 60)} minutos`
    if (diffInSeconds < 86400) return `hace ${Math.floor(diffInSeconds / 3600)} horas`
    if (diffInSeconds < 604800) return `hace ${Math.floor(diffInSeconds / 86400)} días`
    return formatDate(dateObj)
  } else {
    if (diffInSeconds < 60) return 'a few seconds ago'
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`
    return formatDate(dateObj)
  }
}







