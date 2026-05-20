/**
 * Hook para gestión de plantillas de modelos
 * ===========================================
 */

import { useState, useCallback, useMemo } from 'react'

export interface ModelTemplate {
  id: string
  name: string
  description: string
  category: string
  spec: {
    layers: any[]
    optimizer?: string
    loss?: string
    metrics?: string[]
    [key: string]: any
  }
  tags?: string[]
  usage?: string
  examples?: string[]
}

export interface UseModelTemplatesResult {
  templates: ModelTemplate[]
  getTemplate: (id: string) => ModelTemplate | undefined
  getTemplatesByCategory: (category: string) => ModelTemplate[]
  searchTemplates: (query: string) => ModelTemplate[]
  createFromTemplate: (templateId: string, customizations?: Partial<ModelTemplate['spec']>) => ModelTemplate['spec']
  addTemplate: (template: Omit<ModelTemplate, 'id'>) => string
  removeTemplate: (id: string) => void
  categories: string[]
}

/**
 * Plantillas predefinidas
 */
const DEFAULT_TEMPLATES: ModelTemplate[] = [
  {
    id: 'cnn-basic',
    name: 'CNN Básico',
    description: 'Red neuronal convolucional básica para clasificación de imágenes',
    category: 'Computer Vision',
    spec: {
      layers: [
        { type: 'conv2d', params: { filters: 32, kernel_size: 3, activation: 'relu' } },
        { type: 'maxpooling2d', params: { pool_size: 2 } },
        { type: 'conv2d', params: { filters: 64, kernel_size: 3, activation: 'relu' } },
        { type: 'maxpooling2d', params: { pool_size: 2 } },
        { type: 'flatten', params: {} },
        { type: 'dense', params: { units: 128, activation: 'relu' } },
        { type: 'dropout', params: { rate: 0.5 } },
        { type: 'dense', params: { units: 10, activation: 'softmax' } }
      ],
      optimizer: 'adam',
      loss: 'sparsecategoricalcrossentropy',
      metrics: ['accuracy']
    },
    tags: ['cnn', 'image', 'classification'],
    usage: 'Ideal para clasificación de imágenes con CIFAR-10, MNIST, etc.'
  },
  {
    id: 'lstm-text',
    name: 'LSTM para Texto',
    description: 'Modelo LSTM para procesamiento de texto y clasificación',
    category: 'NLP',
    spec: {
      layers: [
        { type: 'embedding', params: { input_dim: 10000, output_dim: 128 } },
        { type: 'lstm', params: { units: 64, return_sequences: false } },
        { type: 'dropout', params: { rate: 0.5 } },
        { type: 'dense', params: { units: 1, activation: 'sigmoid' } }
      ],
      optimizer: 'adam',
      loss: 'binarycrossentropy',
      metrics: ['accuracy']
    },
    tags: ['lstm', 'text', 'nlp'],
    usage: 'Análisis de sentimiento, clasificación de texto'
  },
  {
    id: 'dense-classifier',
    name: 'Clasificador Dense',
    description: 'Red neuronal densa simple para clasificación',
    category: 'General',
    spec: {
      layers: [
        { type: 'dense', params: { units: 128, activation: 'relu' } },
        { type: 'dropout', params: { rate: 0.2 } },
        { type: 'dense', params: { units: 64, activation: 'relu' } },
        { type: 'dropout', params: { rate: 0.2 } },
        { type: 'dense', params: { units: 10, activation: 'softmax' } }
      ],
      optimizer: 'adam',
      loss: 'sparsecategoricalcrossentropy',
      metrics: ['accuracy']
    },
    tags: ['dense', 'classification', 'simple'],
    usage: 'Problemas de clasificación simples con datos tabulares'
  }
]

/**
 * Hook para gestión de plantillas de modelos
 */
export function useModelTemplates(): UseModelTemplatesResult {
  const [templates, setTemplates] = useState<ModelTemplate[]>(DEFAULT_TEMPLATES)

  const getTemplate = useCallback((id: string) => {
    return templates.find(t => t.id === id)
  }, [templates])

  const getTemplatesByCategory = useCallback((category: string) => {
    return templates.filter(t => t.category === category)
  }, [templates])

  const searchTemplates = useCallback((query: string) => {
    const lowerQuery = query.toLowerCase()
    return templates.filter(t =>
      t.name.toLowerCase().includes(lowerQuery) ||
      t.description.toLowerCase().includes(lowerQuery) ||
      t.tags?.some(tag => tag.toLowerCase().includes(lowerQuery)) ||
      t.category.toLowerCase().includes(lowerQuery)
    )
  }, [templates])

  const createFromTemplate = useCallback((
    templateId: string,
    customizations?: Partial<ModelTemplate['spec']>
  ) => {
    const template = getTemplate(templateId)
    if (!template) {
      throw new Error(`Template not found: ${templateId}`)
    }

    return {
      ...template.spec,
      ...customizations,
      layers: customizations?.layers || template.spec.layers
    }
  }, [getTemplate])

  const addTemplate = useCallback((template: Omit<ModelTemplate, 'id'>) => {
    const id = `template-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newTemplate: ModelTemplate = {
      ...template,
      id
    }

    setTemplates(prev => [...prev, newTemplate])
    return id
  }, [])

  const removeTemplate = useCallback((id: string) => {
    setTemplates(prev => prev.filter(t => t.id !== id))
  }, [])

  const categories = useMemo(() => {
    const cats = new Set(templates.map(t => t.category))
    return Array.from(cats).sort()
  }, [templates])

  return {
    templates,
    getTemplate,
    getTemplatesByCategory,
    searchTemplates,
    createFromTemplate,
    addTemplate,
    removeTemplate,
    categories
  }
}

