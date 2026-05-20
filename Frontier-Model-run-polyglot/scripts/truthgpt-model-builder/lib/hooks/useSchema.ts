/**
 * Hook useSchema
 * ==============
 * 
 * Hook para validar datos con esquemas
 */

import { useMemo, useCallback } from 'react'
import { validateSchema, applyDefaults, createSchemaValidator, Schema } from '../utils/schemaUtils'

/**
 * Hook para validar datos con un esquema
 */
export function useSchema<T extends Record<string, any>>(schema: Schema) {
  const validator = useMemo(() => {
    return createSchemaValidator<T>(schema)
  }, [schema])

  const validate = useCallback((data: any) => {
    return validator(data)
  }, [validator])

  const applySchemaDefaults = useCallback((data: any): T => {
    return applyDefaults(data, schema) as T
  }, [schema])

  return {
    validate,
    applyDefaults: applySchemaDefaults,
    schema
  }
}






