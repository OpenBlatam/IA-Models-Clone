/**
 * Model Templates
 * Plantillas predefinidas de modelos para construcción rápida
 */

export interface ModelTemplate {
  id: string
  name: string
  description: string
  category: 'nlp' | 'vision' | 'audio' | 'multimodal' | 'generative' | 'classification'
  tags: string[]
  priority: number
  config?: {
    architecture?: string
    layers?: number[]
    optimizer?: string
    learningRate?: number
    epochs?: number
  }
}

export const MODEL_TEMPLATES: ModelTemplate[] = [
  {
    id: 'sentiment-analysis',
    name: 'Análisis de Sentimientos',
    description: 'Modelo para análisis de sentimientos en texto en español',
    category: 'nlp',
    tags: ['sentiment', 'text', 'classification', 'spanish'],
    priority: 5,
    config: {
      architecture: 'transformer',
      layers: [512, 256, 128],
      optimizer: 'adamw',
      learningRate: 2e-5,
      epochs: 5,
    },
  },
  {
    id: 'image-classifier',
    name: 'Clasificador de Imágenes',
    description: 'Modelo CNN para clasificación de imágenes',
    category: 'vision',
    tags: ['vision', 'cnn', 'classification', 'images'],
    priority: 5,
    config: {
      architecture: 'cnn',
      layers: [64, 128, 256],
      optimizer: 'adamw',
      learningRate: 1e-4,
      epochs: 10,
    },
  },
  {
    id: 'text-generator',
    name: 'Generador de Texto',
    description: 'Modelo generativo de texto basado en GPT',
    category: 'generative',
    tags: ['generative', 'text', 'gpt', 'llm'],
    priority: 7,
    config: {
      architecture: 'transformer',
      layers: [1024, 512, 256],
      optimizer: 'adamw',
      learningRate: 5e-5,
      epochs: 3,
    },
  },
  {
    id: 'translation-model',
    name: 'Modelo de Traducción',
    description: 'Modelo para traducción automática entre idiomas',
    category: 'nlp',
    tags: ['translation', 'nlp', 'multilingual'],
    priority: 6,
    config: {
      architecture: 'transformer',
      layers: [512, 512],
      optimizer: 'adamw',
      learningRate: 3e-5,
      epochs: 5,
    },
  },
  {
    id: 'object-detection',
    name: 'Detección de Objetos',
    description: 'Modelo para detección de objetos en imágenes',
    category: 'vision',
    tags: ['vision', 'detection', 'yolo', 'objects'],
    priority: 8,
    config: {
      architecture: 'cnn',
      layers: [128, 256, 512],
      optimizer: 'adamw',
      learningRate: 1e-4,
      epochs: 15,
    },
  },
  {
    id: 'speech-recognition',
    name: 'Reconocimiento de Voz',
    description: 'Modelo para reconocimiento de voz y transcripción',
    category: 'audio',
    tags: ['audio', 'speech', 'asr', 'transcription'],
    priority: 7,
    config: {
      architecture: 'lstm',
      layers: [256, 128],
      optimizer: 'adamw',
      learningRate: 2e-4,
      epochs: 10,
    },
  },
  {
    id: 'multimodal-vqa',
    name: 'Visual Question Answering',
    description: 'Modelo multimodal para responder preguntas sobre imágenes',
    category: 'multimodal',
    tags: ['multimodal', 'vision', 'nlp', 'vqa'],
    priority: 9,
    config: {
      architecture: 'transformer',
      layers: [512, 512, 256],
      optimizer: 'adamw',
      learningRate: 5e-5,
      epochs: 8,
    },
  },
  {
    id: 'text-classifier',
    name: 'Clasificador de Texto',
    description: 'Modelo para clasificación de texto en múltiples categorías',
    category: 'classification',
    tags: ['classification', 'text', 'nlp'],
    priority: 4,
    config: {
      architecture: 'dense',
      layers: [256, 128, 64],
      optimizer: 'adam',
      learningRate: 1e-3,
      epochs: 5,
    },
  },
]

/**
 * Obtener plantillas por categoría
 */
export function getTemplatesByCategory(category: string): ModelTemplate[] {
  return MODEL_TEMPLATES.filter(template => template.category === category)
}

/**
 * Obtener plantillas por tag
 */
export function getTemplatesByTag(tag: string): ModelTemplate[] {
  return MODEL_TEMPLATES.filter(template => template.tags.includes(tag))
}

/**
 * Buscar plantillas
 */
export function searchTemplates(query: string): ModelTemplate[] {
  const lowerQuery = query.toLowerCase()
  return MODEL_TEMPLATES.filter(template =>
    template.name.toLowerCase().includes(lowerQuery) ||
    template.description.toLowerCase().includes(lowerQuery) ||
    template.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
  )
}

/**
 * Obtener plantilla por ID
 */
export function getTemplateById(id: string): ModelTemplate | undefined {
  return MODEL_TEMPLATES.find(template => template.id === id)
}

/**
 * Obtener todas las categorías
 */
export function getAllCategories(): string[] {
  return Array.from(new Set(MODEL_TEMPLATES.map(t => t.category)))
}

/**
 * Obtener todos los tags
 */
export function getAllTags(): string[] {
  const allTags = MODEL_TEMPLATES.flatMap(t => t.tags)
  return Array.from(new Set(allTags)).sort()
}










