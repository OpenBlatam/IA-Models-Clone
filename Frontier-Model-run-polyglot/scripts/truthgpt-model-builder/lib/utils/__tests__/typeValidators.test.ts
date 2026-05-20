/**
 * Tests para Validadores de Tipos
 * ================================
 */

import {
  isValidModelStatus,
  isValidModelType,
  isValidArchitecture,
  isValidOptimizer,
  isValidLossFunction,
  isValidMetric,
  isValidLayerType,
  validateModelSpec,
  validateModelState,
  validateFields
} from '../typeValidators'
import { ModelSpec, ModelState } from '../../types/modelTypes'

describe('Type Validators', () => {
  describe('isValidModelStatus', () => {
    it('debe validar estados válidos', () => {
      expect(isValidModelStatus('creating')).toBe(true)
      expect(isValidModelStatus('completed')).toBe(true)
      expect(isValidModelStatus('failed')).toBe(true)
    })

    it('debe rechazar estados inválidos', () => {
      expect(isValidModelStatus('invalid')).toBe(false)
      expect(isValidModelStatus('')).toBe(false)
      expect(isValidModelStatus(null)).toBe(false)
      expect(isValidModelStatus(undefined)).toBe(false)
    })
  })

  describe('isValidModelType', () => {
    it('debe validar tipos válidos', () => {
      expect(isValidModelType('classification')).toBe(true)
      expect(isValidModelType('regression')).toBe(true)
      expect(isValidModelType('nlp')).toBe(true)
    })

    it('debe rechazar tipos inválidos', () => {
      expect(isValidModelType('invalid')).toBe(false)
      expect(isValidModelType('')).toBe(false)
    })
  })

  describe('validateModelSpec', () => {
    it('debe validar un spec válido', () => {
      const spec: ModelSpec = {
        type: 'classification',
        architecture: 'dense',
        layers: [
          {
            type: 'Dense',
            params: { units: 64 }
          }
        ],
        optimizer: {
          type: 'adam',
          learningRate: 0.001
        },
        loss: {
          type: 'categorical_crossentropy'
        },
        metrics: ['accuracy']
      }

      const result = validateModelSpec(spec)
      expect(result.valid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('debe rechazar un spec sin tipo', () => {
      const spec = {
        architecture: 'dense',
        layers: []
      }

      const result = validateModelSpec(spec)
      expect(result.valid).toBe(false)
      expect(result.errors.length).toBeGreaterThan(0)
    })

    it('debe rechazar un spec sin capas', () => {
      const spec: Partial<ModelSpec> = {
        type: 'classification',
        architecture: 'dense',
        layers: [],
        optimizer: { type: 'adam' },
        loss: { type: 'categorical_crossentropy' },
        metrics: []
      }

      const result = validateModelSpec(spec)
      expect(result.valid).toBe(false)
      expect(result.errors.some(e => e.field === 'layers')).toBe(true)
    })

    it('debe generar advertencias para specs con muchas capas', () => {
      const spec: ModelSpec = {
        type: 'classification',
        architecture: 'dense',
        layers: Array(15).fill({
          type: 'Dense',
          params: { units: 64 }
        }),
        optimizer: { type: 'adam' },
        loss: { type: 'categorical_crossentropy' },
        metrics: []
      }

      const result = validateModelSpec(spec)
      expect(result.warnings.length).toBeGreaterThan(0)
    })
  })

  describe('validateModelState', () => {
    it('debe validar un state válido', () => {
      const state: ModelState = {
        id: 'model-123',
        name: 'Test Model',
        status: 'creating',
        progress: 50,
        createdAt: new Date(),
        updatedAt: new Date()
      }

      const result = validateModelState(state)
      expect(result.valid).toBe(true)
    })

    it('debe rechazar un state sin id', () => {
      const state = {
        name: 'Test Model',
        status: 'creating',
        progress: 50,
        createdAt: new Date(),
        updatedAt: new Date()
      }

      const result = validateModelState(state)
      expect(result.valid).toBe(false)
      expect(result.errors.some(e => e.field === 'id')).toBe(true)
    })

    it('debe rechazar un state con progreso inválido', () => {
      const state: Partial<ModelState> = {
        id: 'model-123',
        name: 'Test Model',
        status: 'creating',
        progress: 150, // Inválido
        createdAt: new Date(),
        updatedAt: new Date()
      }

      const result = validateModelState(state)
      expect(result.valid).toBe(false)
      expect(result.errors.some(e => e.field === 'progress')).toBe(true)
    })
  })

  describe('validateFields', () => {
    it('debe validar múltiples campos', () => {
      const obj = {
        name: 'Test',
        age: 25
      }

      const validators = {
        name: (v: unknown) => typeof v === 'string' && v.length > 0,
        age: (v: unknown) => typeof v === 'number' && v > 0
      }

      const result = validateFields(obj, validators)
      expect(result.valid).toBe(true)
    })

    it('debe detectar campos inválidos', () => {
      const obj = {
        name: '',
        age: -5
      }

      const validators = {
        name: (v: unknown) => typeof v === 'string' && v.length > 0,
        age: (v: unknown) => typeof v === 'number' && v > 0
      }

      const result = validateFields(obj, validators)
      expect(result.valid).toBe(false)
      expect(result.errors.length).toBe(2)
    })
  })
})







