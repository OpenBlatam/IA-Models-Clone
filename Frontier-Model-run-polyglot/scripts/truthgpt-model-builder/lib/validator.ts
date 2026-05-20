/**
 * Input validation utilities
 */

export interface ValidationResult {
  isValid: boolean
  message: string
  type: 'success' | 'warning' | 'error' | 'info'
}

export function validateDescription(description: string): ValidationResult {
  if (!description.trim()) {
    return {
      isValid: false,
      message: 'La descripción no puede estar vacía',
      type: 'error',
    }
  }

  if (description.length < 10) {
    return {
      isValid: false,
      message: 'La descripción es muy corta. Sé más específico.',
      type: 'warning',
    }
  }

  if (description.length > 500) {
    return {
      isValid: false,
      message: 'La descripción es muy larga. Intenta ser más conciso.',
      type: 'warning',
    }
  }

  // Check for minimum keywords
  const keywords = ['modelo', 'para', 'análisis', 'clasificar', 'predecir', 'generar', 'detectar']
  const hasKeywords = keywords.some(kw => description.toLowerCase().includes(kw))

  if (!hasKeywords) {
    return {
      isValid: true,
      message: '💡 Tip: Intenta ser más específico sobre qué tipo de modelo necesitas',
      type: 'info',
    }
  }

  // Check for good description patterns
  const goodPatterns = [
    /\b(para|que|capaz de|que pueda)\b/i,
    /\b(clasificar|predecir|generar|detectar|analizar|reconocer)\b/i,
  ]

  const patternMatches = goodPatterns.filter(pattern => pattern.test(description)).length

  if (patternMatches >= 2) {
    return {
      isValid: true,
      message: '✅ Descripción válida. ¡El modelo se creará perfectamente!',
      type: 'success',
    }
  }

  return {
    isValid: true,
    message: '💡 Buena descripción. Puedes agregar más detalles para mejores resultados.',
    type: 'info',
  }
}

export function validateModelName(name: string): ValidationResult {
  if (!name.trim()) {
    return {
      isValid: false,
      message: 'El nombre del modelo es requerido',
      type: 'error',
    }
  }

  if (name.length < 3) {
    return {
      isValid: false,
      message: 'El nombre debe tener al menos 3 caracteres',
      type: 'error',
    }
  }

  if (!/^[a-z0-9-]+$/.test(name)) {
    return {
      isValid: false,
      message: 'El nombre solo puede contener letras minúsculas, números y guiones',
      type: 'error',
    }
  }

  return {
    isValid: true,
    message: 'Nombre válido',
    type: 'success',
  }
}


