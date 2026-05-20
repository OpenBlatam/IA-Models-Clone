/**
 * Hook useForm Mejorado
 * =====================
 * 
 * Hook para manejar formularios con validación
 */

import { useState, useCallback, useRef } from 'react'

export interface ValidationRule<T = unknown> {
  validator: (value: T) => boolean
  message: string
}

export interface FieldConfig<T = unknown> {
  initialValue?: T
  rules?: ValidationRule<T>[]
  required?: boolean
  requiredMessage?: string
}

export interface UseFormOptions<T extends Record<string, unknown>> {
  initialValues?: Partial<T>
  onSubmit?: (values: T) => void | Promise<void>
  validateOnChange?: boolean
  validateOnBlur?: boolean
}

export interface FieldState<T = unknown> {
  value: T
  error: string | null
  touched: boolean
  dirty: boolean
}

export interface UseFormResult<T extends Record<string, unknown>> {
  values: T
  errors: Partial<Record<keyof T, string>>
  touched: Partial<Record<keyof T, boolean>>
  dirty: Partial<Record<keyof T, boolean>>
  isValid: boolean
  isSubmitting: boolean
  setValue: <K extends keyof T>(field: K, value: T[K]) => void
  setError: <K extends keyof T>(field: K, error: string | null) => void
  setTouched: <K extends keyof T>(field: K, touched: boolean) => void
  getFieldProps: <K extends keyof T>(field: K) => {
    value: T[K]
    onChange: (value: T[K]) => void
    onBlur: () => void
    error: string | null
  }
  handleSubmit: (e?: React.FormEvent) => Promise<void>
  reset: () => void
  validate: () => boolean
  validateField: <K extends keyof T>(field: K) => boolean
}

/**
 * Hook para manejar formularios
 */
export function useForm<T extends Record<string, unknown>>(
  options: UseFormOptions<T> = {}
): UseFormResult<T> {
  const {
    initialValues = {} as Partial<T>,
    onSubmit,
    validateOnChange = false,
    validateOnBlur = true
  } = options

  const [values, setValues] = useState<T>(initialValues as T)
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({})
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({})
  const [dirty, setDirty] = useState<Partial<Record<keyof T, boolean>>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  const fieldConfigsRef = useRef<Partial<Record<keyof T, FieldConfig>>>({})

  const setValue = useCallback(<K extends keyof T>(field: K, value: T[K]) => {
    setValues(prev => ({ ...prev, [field]: value }))
    setDirty(prev => ({ ...prev, [field]: true }))
    
    if (validateOnChange) {
      validateField(field)
    }
  }, [validateOnChange])

  const setError = useCallback(<K extends keyof T>(field: K, error: string | null) => {
    setErrors(prev => {
      if (error === null) {
        const { [field]: _, ...rest } = prev
        return rest
      }
      return { ...prev, [field]: error }
    })
  }, [])

  const setTouched = useCallback(<K extends keyof T>(field: K, touchedValue: boolean) => {
    setTouched(prev => ({ ...prev, [field]: touchedValue }))
  }, [])

  const validateField = useCallback(<K extends keyof T>(field: K): boolean => {
    const config = fieldConfigsRef.current[field]
    if (!config) return true

    const value = values[field]

    // Validar required
    if (config.required && (value === null || value === undefined || value === '')) {
      setError(field, config.requiredMessage || `${String(field)} es requerido`)
      return false
    }

    // Validar reglas
    if (config.rules) {
      for (const rule of config.rules) {
        if (!rule.validator(value as unknown)) {
          setError(field, rule.message)
          return false
        }
      }
    }

    setError(field, null)
    return true
  }, [values])

  const validate = useCallback((): boolean => {
    let isValid = true

    for (const field in fieldConfigsRef.current) {
      if (!validateField(field)) {
        isValid = false
      }
    }

    return isValid
  }, [validateField])

  const getFieldProps = useCallback(<K extends keyof T>(field: K) => {
    return {
      value: values[field],
      onChange: (value: T[K]) => setValue(field, value),
      onBlur: () => {
        setTouched(field, true)
        if (validateOnBlur) {
          validateField(field)
        }
      },
      error: errors[field] || null
    }
  }, [values, errors, setValue, setTouched, validateOnBlur, validateField])

  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault()

    // Marcar todos los campos como touched
    const allFields = Object.keys(fieldConfigsRef.current) as Array<keyof T>
    allFields.forEach(field => setTouched(field, true))

    // Validar
    if (!validate()) {
      return
    }

    setIsSubmitting(true)
    try {
      await onSubmit?.(values)
    } catch (error) {
      console.error('Form submission error:', error)
    } finally {
      setIsSubmitting(false)
    }
  }, [values, onSubmit, validate, setTouched])

  const reset = useCallback(() => {
    setValues(initialValues as T)
    setErrors({})
    setTouched({})
    setDirty({})
  }, [initialValues])

  const isValid = Object.keys(errors).length === 0

  return {
    values,
    errors,
    touched,
    dirty,
    isValid,
    isSubmitting,
    setValue,
    setError,
    setTouched,
    getFieldProps,
    handleSubmit,
    reset,
    validate,
    validateField
  }
}

/**
 * Registra configuración de campo
 */
export function registerField<T extends Record<string, unknown>, K extends keyof T>(
  form: UseFormResult<T>,
  field: K,
  config: FieldConfig<T[K]>
): void {
  // Esta función se puede usar para registrar campos dinámicamente
  // La implementación real dependería de cómo se almacene la configuración
}







