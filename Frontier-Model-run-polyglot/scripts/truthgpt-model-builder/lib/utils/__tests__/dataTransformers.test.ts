/**
 * Tests para Transformadores de Datos
 * ====================================
 */

import {
  transformSpecToAPI,
  transformAPIToSpec,
  transformStateToReadable,
  createModelState,
  normalizeLayerConfig,
  normalizeLayers,
  normalizeOptimizerConfig,
  normalizeLossConfig,
  serializeModelState,
  deserializeModelState
} from '../dataTransformers'
import { ModelSpec, ModelState } from '../../types/modelTypes'

describe('Data Transformers', () => {
  describe('transformSpecToAPI', () => {
    it('debe transformar un spec a formato API', () => {
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

      const apiFormat = transformSpecToAPI(spec)
      expect(apiFormat.type).toBe('classification')
      expect(apiFormat.architecture).toBe('dense')
      expect(apiFormat.layers).toHaveLength(1)
      expect(apiFormat.optimizer.type).toBe('adam')
      expect(apiFormat.optimizer.learning_rate).toBe(0.001)
    })
  })

  describe('transformAPIToSpec', () => {
    it('debe transformar formato API a spec', () => {
      const apiData = {
        type: 'classification',
        architecture: 'dense',
        layers: [
          {
            type: 'Dense',
            units: 64
          }
        ],
        optimizer: {
          type: 'adam',
          learning_rate: 0.001
        },
        loss: {
          type: 'categorical_crossentropy'
        },
        metrics: ['accuracy']
      }

      const spec = transformAPIToSpec(apiData)
      expect(spec.type).toBe('classification')
      expect(spec.architecture).toBe('dense')
      expect(spec.layers).toHaveLength(1)
      expect(spec.optimizer.type).toBe('adam')
      expect(spec.optimizer.learningRate).toBe(0.001)
    })
  })

  describe('transformStateToReadable', () => {
    it('debe transformar state a formato legible', () => {
      const state: ModelState = {
        id: 'model-123',
        name: 'Test Model',
        status: 'creating',
        progress: 50,
        createdAt: new Date('2024-01-01'),
        updatedAt: new Date('2024-01-01')
      }

      const readable = transformStateToReadable(state)
      expect(readable.id).toBe('model-123')
      expect(readable.name).toBe('Test Model')
      expect(readable.status).toBe('Creando')
      expect(readable.progress).toBe('50%')
    })
  })

  describe('createModelState', () => {
    it('debe crear un ModelState desde datos parciales', () => {
      const state = createModelState({
        id: 'model-123',
        name: 'Test Model',
        status: 'creating',
        progress: 50
      })

      expect(state.id).toBe('model-123')
      expect(state.name).toBe('Test Model')
      expect(state.status).toBe('creating')
      expect(state.progress).toBe(50)
      expect(state.createdAt).toBeInstanceOf(Date)
      expect(state.updatedAt).toBeInstanceOf(Date)
    })
  })

  describe('normalizeLayerConfig', () => {
    it('debe normalizar una configuración de capa válida', () => {
      const layer = {
        type: 'Dense',
        params: { units: 64 }
      }

      const normalized = normalizeLayerConfig(layer)
      expect(normalized).not.toBeNull()
      expect(normalized?.type).toBe('Dense')
      expect(normalized?.params.units).toBe(64)
    })

    it('debe retornar null para datos inválidos', () => {
      expect(normalizeLayerConfig(null)).toBeNull()
      expect(normalizeLayerConfig(undefined)).toBeNull()
      expect(normalizeLayerConfig({})).toBeNull()
    })
  })

  describe('normalizeLayers', () => {
    it('debe normalizar un array de capas', () => {
      const layers = [
        { type: 'Dense', params: { units: 64 } },
        { type: 'Dropout', params: { rate: 0.5 } },
        null, // Debe ser filtrado
        { type: 'Dense', params: { units: 32 } }
      ]

      const normalized = normalizeLayers(layers)
      expect(normalized).toHaveLength(3)
      expect(normalized[0].type).toBe('Dense')
      expect(normalized[1].type).toBe('Dropout')
    })
  })

  describe('serializeModelState / deserializeModelState', () => {
    it('debe serializar y deserializar correctamente', () => {
      const state: ModelState = {
        id: 'model-123',
        name: 'Test Model',
        status: 'creating',
        progress: 50,
        createdAt: new Date('2024-01-01'),
        updatedAt: new Date('2024-01-01')
      }

      const serialized = serializeModelState(state)
      const deserialized = deserializeModelState(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.id).toBe(state.id)
      expect(deserialized?.name).toBe(state.name)
      expect(deserialized?.status).toBe(state.status)
      expect(deserialized?.progress).toBe(state.progress)
      expect(deserialized?.createdAt).toBeInstanceOf(Date)
    })

    it('debe retornar null para JSON inválido', () => {
      expect(deserializeModelState('invalid json')).toBeNull()
    })
  })
})







