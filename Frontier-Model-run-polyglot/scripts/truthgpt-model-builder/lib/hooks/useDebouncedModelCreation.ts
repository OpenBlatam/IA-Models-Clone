/**
 * Hook para creación de modelos con debounce
 * ===========================================
 * 
 * Previene múltiples creaciones accidentales y optimiza validación
 */

import { useState, useCallback, useRef } from 'react'
import { useDebouncedCallback } from '../optimization-utils'

export interface UseDebouncedModelCreationOptions {
  debounceDelay?: number
  onValidationChange?: (isValid: boolean, errors: string[]) => void
}

export interface UseDebouncedModelCreationResult {
  validateDescription: (description: string) => void
  isValidationPending: boolean
  validationResult: {
    isValid: boolean
    errors: string[]
  } | null
  clearValidation: () => void
}

/**
 * Hook para validación con debounce antes de crear modelos
 */
export function useDebouncedModelCreation(
  options: UseDebouncedModelCreationOptions = {}
): UseDebouncedModelCreationResult {
  const { debounceDelay = 500, onValidationChange } = options
  const [isValidationPending, setIsValidationPending] = useState(false)
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean
    errors: string[]
  } | null>(null)
  const validationAbortRef = useRef<AbortController | null>(null)

  const performValidation = useCallback(async (description: string) => {
    // Cancelar validación anterior si existe
    if (validationAbortRef.current) {
      validationAbortRef.current.abort()
    }

    validationAbortRef.current = new AbortController()
    const signal = validationAbortRef.current.signal

    setIsValidationPending(true)

    try {
      // Validación básica
      const trimmed = description.trim()
      const errors: string[] = []

      if (!trimmed || trimmed.length === 0) {
        errors.push('La descripción no puede estar vacía')
      } else {
        if (trimmed.length < 10) {
          errors.push('La descripción debe tener al menos 10 caracteres')
        }
        if (trimmed.length > 5000) {
          errors.push('La descripción es demasiado larga (máximo 5000 caracteres)')
        }
      }

      // Verificar si fue cancelado
      if (signal.aborted) {
        return
      }

      const result = {
        isValid: errors.length === 0,
        errors
      }

      setValidationResult(result)
      onValidationChange?.(result.isValid, result.errors)

      return result
    } catch (error) {
      if (!signal.aborted) {
        console.error('Error en validación:', error)
        const result = {
          isValid: false,
          errors: ['Error al validar la descripción']
        }
        setValidationResult(result)
        onValidationChange?.(false, result.errors)
      }
    } finally {
      if (!signal.aborted) {
        setIsValidationPending(false)
        validationAbortRef.current = null
      }
    }
  }, [onValidationChange])

  const debouncedValidation = useDebouncedCallback(
    performValidation,
    debounceDelay
  )

  const validateDescription = useCallback(
    (description: string) => {
      if (!description || description.trim().length === 0) {
        setValidationResult(null)
        onValidationChange?.(true, [])
        return
      }

      debouncedValidation(description)
    },
    [debouncedValidation, onValidationChange]
  )

  const clearValidation = useCallback(() => {
    if (validationAbortRef.current) {
      validationAbortRef.current.abort()
      validationAbortRef.current = null
    }
    setIsValidationPending(false)
    setValidationResult(null)
  }, [])

  return {
    validateDescription,
    isValidationPending,
    validationResult,
    clearValidation
  }
}

