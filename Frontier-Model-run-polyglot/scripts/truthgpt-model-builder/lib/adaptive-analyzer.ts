/**
 * Adaptive Model Analyzer - Advanced AI-powered model architecture analysis
 * Adapts to user descriptions with context understanding and pattern recognition
 */

import { ModelSpec } from './model-analyzer'

interface Context {
  domain?: string
  complexity?: 'simple' | 'medium' | 'complex' | 'very-complex'
  dataSize?: 'small' | 'medium' | 'large' | 'very-large'
  performance?: 'speed' | 'accuracy' | 'balanced'
  constraints?: string[]
}

export function adaptiveAnalyze(description: string, context?: Context): ModelSpec {
  const lowerDesc = description.toLowerCase()
  
  // Extract context from description
  const extractedContext = extractContext(description)
  const finalContext = { ...extractedContext, ...context }
  
  // Multi-factor analysis
  const type = determineType(lowerDesc, finalContext)
  const architecture = determineArchitecture(lowerDesc, type, finalContext)
  const complexity = determineComplexity(lowerDesc, finalContext)
  
  // Generate adaptive specifications
  const layers = generateAdaptiveLayers(architecture, complexity, type, finalContext)
  const optimizer = determineOptimizer(lowerDesc, type, complexity, finalContext)
  const learningRate = determineLearningRate(type, architecture, complexity, finalContext)
  const batchSize = determineBatchSize(complexity, finalContext)
  const epochs = determineEpochs(type, complexity, finalContext)
  const dropout = determineDropout(architecture, complexity, finalContext)
  
  return {
    type,
    architecture,
    layers,
    activation: determineActivation(architecture, type, finalContext),
    outputActivation: determineOutputActivation(type),
    useDropout: dropout > 0,
    dropoutRate: dropout,
    useBatchNorm: architecture === 'cnn' || (complexity === 'complex' && architecture !== 'transformer'),
    optimizer,
    learningRate,
    batchSize,
    epochs,
    metrics: determineMetrics(type),
    loss: determineLoss(type),
  }
}

function extractContext(description: string): Context {
  const lower = description.toLowerCase()
  const context: Context = {}
  
  // Detect domain
  const domains = {
    healthcare: ['salud', 'médico', 'medical', 'health', 'paciente', 'diagnóstico', 'diagnosis'],
    finance: ['finanzas', 'finance', 'bancario', 'bank', 'trading', 'inversión', 'investment'],
    ecommerce: ['ecommerce', 'tienda', 'shop', 'producto', 'product', 'venta', 'sale'],
    education: ['educación', 'education', 'estudiante', 'student', 'aprendizaje', 'learning'],
    social: ['red social', 'social', 'comunidad', 'community', 'usuario', 'user'],
  }
  
  for (const [domain, keywords] of Object.entries(domains)) {
    if (keywords.some(kw => lower.includes(kw))) {
      context.domain = domain
      break
    }
  }
  
  // Detect complexity indicators
  if (lower.includes('simple') || lower.includes('básico') || lower.includes('básica')) {
    context.complexity = 'simple'
  } else if (lower.includes('complejo') || lower.includes('complex') || lower.includes('avanzado')) {
    context.complexity = 'complex'
  } else if (lower.includes('muy complejo') || lower.includes('very complex')) {
    context.complexity = 'very-complex'
  } else {
    context.complexity = 'medium'
  }
  
  // Detect data size
  if (lower.includes('grande') || lower.includes('large') || lower.includes('millones') || lower.includes('millions')) {
    context.dataSize = 'large'
  } else if (lower.includes('pequeño') || lower.includes('small') || lower.includes('miles') || lower.includes('thousands')) {
    context.dataSize = 'small'
  } else if (lower.includes('muy grande') || lower.includes('very large')) {
    context.dataSize = 'very-large'
  } else {
    context.dataSize = 'medium'
  }
  
  // Detect performance preference
  if (lower.includes('rápido') || lower.includes('fast') || lower.includes('velocidad') || lower.includes('speed')) {
    context.performance = 'speed'
  } else if (lower.includes('precisión') || lower.includes('accuracy') || lower.includes('exactitud')) {
    context.performance = 'accuracy'
  } else {
    context.performance = 'balanced'
  }
  
  // Extract constraints
  context.constraints = []
  if (lower.includes('memoria') || lower.includes('memory') || lower.includes('ligero') || lower.includes('lightweight')) {
    context.constraints.push('memory')
  }
  if (lower.includes('tiempo real') || lower.includes('real-time') || lower.includes('tiempo real')) {
    context.constraints.push('realtime')
  }
  if (lower.includes('móvil') || lower.includes('mobile') || lower.includes('edge')) {
    context.constraints.push('mobile')
  }
  
  return context
}

function determineType(description: string, context: Context): ModelSpec['type'] {
  // Enhanced keyword matching with context
  const typeScores: Record<string, number> = {
    classification: 0,
    regression: 0,
    nlp: 0,
    vision: 0,
    'time-series': 0,
    generative: 0,
  }
  
  // Classification keywords
  const classificationKw = ['clasificar', 'classify', 'categoría', 'category', 'etiqueta', 'label', 'spam', 'fraude', 'fraud']
  classificationKw.forEach(kw => {
    if (description.includes(kw)) typeScores.classification += 2
  })
  
  // Regression keywords
  const regressionKw = ['predecir', 'predict', 'regresión', 'regression', 'valor', 'value', 'precio', 'price', 'cantidad']
  regressionKw.forEach(kw => {
    if (description.includes(kw)) typeScores.regression += 2
  })
  
  // NLP keywords
  const nlpKw = ['texto', 'text', 'nlp', 'lenguaje', 'language', 'sentimiento', 'sentiment', 'análisis', 'analysis', 'traducción', 'translation', 'resumen', 'summary']
  nlpKw.forEach(kw => {
    if (description.includes(kw)) typeScores.nlp += 2
  })
  
  // Vision keywords
  const visionKw = ['imagen', 'image', 'imágenes', 'images', 'visual', 'reconocimiento', 'recognition', 'detección', 'detection', 'cámara', 'camera', 'foto', 'photo']
  visionKw.forEach(kw => {
    if (description.includes(kw)) typeScores.vision += 2
  })
  
  // Time-series keywords
  const timeSeriesKw = ['tiempo', 'time', 'serie', 'series', 'temporal', 'secuencia', 'sequence', 'predicción', 'forecast', 'tendencia', 'trend']
  timeSeriesKw.forEach(kw => {
    if (description.includes(kw)) typeScores['time-series'] += 2
  })
  
  // Generative keywords
  const generativeKw = ['generar', 'generate', 'crear', 'create', 'generativo', 'generative', 'gpt', 'llm', 'lenguaje model', 'texto automático']
  generativeKw.forEach(kw => {
    if (description.includes(kw)) typeScores.generative += 3 // Higher weight
  })
  
  // Context-based adjustments
  if (context.domain === 'healthcare' && description.includes('imagen')) {
    typeScores.vision += 3
  }
  if (context.domain === 'finance' && description.includes('predecir')) {
    typeScores.regression += 2
    typeScores['time-series'] += 2
  }
  
  // Find max score
  const maxType = Object.entries(typeScores).reduce((a, b) => typeScores[a[0]] > typeScores[b[0]] ? a : b)[0]
  
  return (maxType || 'custom') as ModelSpec['type']
}

function determineArchitecture(description: string, type: ModelSpec['type'], context: Context): ModelSpec['architecture'] {
  // Explicit architecture mentions
  if (description.includes('transformer') || description.includes('gpt') || description.includes('llm') || description.includes('bert')) {
    return 'transformer'
  }
  if (description.includes('cnn') || description.includes('convolucional') || description.includes('convolutional')) {
    return 'cnn'
  }
  if (description.includes('lstm') || description.includes('rnn') || description.includes('recurrent')) {
    return 'lstm'
  }
  
  // Type-based architecture
  if (type === 'vision') {
    return 'cnn'
  }
  if (type === 'nlp' || type === 'generative') {
    // Prefer transformer for complex NLP, LSTM for simpler
    if (context.complexity === 'complex' || context.complexity === 'very-complex') {
      return 'transformer'
    }
    return 'lstm'
  }
  if (type === 'time-series') {
    return 'lstm'
  }
  
  // Context-based decisions
  if (context.constraints?.includes('mobile') || context.constraints?.includes('memory')) {
    return 'dense' // Lighter architecture
  }
  
  if (context.performance === 'speed') {
    return 'dense' // Faster inference
  }
  
  if (context.performance === 'accuracy' && type === 'nlp') {
    return 'transformer'
  }
  
  return 'dense'
}

function determineComplexity(description: string, context: Context): 'simple' | 'medium' | 'complex' | 'very-complex' {
  if (context.complexity) {
    return context.complexity
  }
  
  // Analyze description length and keywords
  const complexityKeywords = {
    simple: ['simple', 'básico', 'pequeño', 'small', 'fácil', 'easy'],
    complex: ['complejo', 'complex', 'avanzado', 'advanced', 'sophisticated'],
    'very-complex': ['muy complejo', 'very complex', 'extremadamente', 'extremely', 'deep', 'profundo'],
  }
  
  let score = 0
  Object.entries(complexityKeywords.simple).forEach(kw => {
    if (description.includes(kw)) score -= 1
  })
  Object.entries(complexityKeywords.complex).forEach(kw => {
    if (description.includes(kw)) score += 1
  })
  Object.entries(complexityKeywords['very-complex']).forEach(kw => {
    if (description.includes(kw)) score += 2
  })
  
  if (score <= -1) return 'simple'
  if (score >= 2) return 'very-complex'
  if (score >= 1) return 'complex'
  return 'medium'
}

function generateAdaptiveLayers(
  architecture: ModelSpec['architecture'],
  complexity: 'simple' | 'medium' | 'complex' | 'very-complex',
  type: ModelSpec['type'],
  context: Context
): number[] {
  const baseLayers: Record<string, number[]> = {
    dense: [256, 128, 64],
    cnn: [64, 128, 256, 128, 64],
    lstm: [128, 64, 32],
    transformer: [512, 256, 128],
  }
  
  let layers = [...baseLayers[architecture] || baseLayers.dense]
  
  // Adjust based on complexity
  if (complexity === 'simple') {
    layers = layers.slice(0, 2).map(l => Math.floor(l * 0.7))
  } else if (complexity === 'medium') {
    // Keep base layers
  } else if (complexity === 'complex') {
    layers = layers.map(l => Math.floor(l * 1.3))
    layers.push(Math.floor(layers[layers.length - 1] * 0.8))
  } else if (complexity === 'very-complex') {
    layers = layers.map(l => Math.floor(l * 1.5))
    layers.push(Math.floor(layers[layers.length - 1] * 0.9))
    layers.push(Math.floor(layers[layers.length - 1] * 0.7))
  }
  
  // Adjust based on data size
  if (context.dataSize === 'large' || context.dataSize === 'very-large') {
    layers = layers.map(l => Math.floor(l * 1.2))
  } else if (context.dataSize === 'small') {
    layers = layers.map(l => Math.floor(l * 0.8))
  }
  
  // Adjust based on constraints
  if (context.constraints?.includes('memory') || context.constraints?.includes('mobile')) {
    layers = layers.map(l => Math.floor(l * 0.6))
    layers = layers.slice(0, 3) // Limit layers
  }
  
  // Ensure minimum sizes
  layers = layers.map(l => Math.max(l, 32))
  
  return layers
}

function determineOptimizer(
  description: string,
  type: ModelSpec['type'],
  complexity: string,
  context: Context
): ModelSpec['optimizer'] {
  // Explicit mentions
  if (description.includes('adamw') || description.includes('adam w')) return 'adamw'
  if (description.includes('sgd')) return 'sgd'
  if (description.includes('rmsprop')) return 'rmsprop'
  
  // Type-based defaults
  if (type === 'generative' || type === 'nlp') {
    return 'adamw' // Better for transformers
  }
  
  if (type === 'vision' && complexity === 'complex') {
    return 'adamw'
  }
  
  if (context.performance === 'speed') {
    return 'sgd' // Faster convergence sometimes
  }
  
  return 'adam' // Default
}

function determineLearningRate(
  type: ModelSpec['type'],
  architecture: ModelSpec['architecture'],
  complexity: string,
  context: Context
): number {
  // Base learning rates
  const baseLR: Record<string, number> = {
    generative: 0.0001,
    nlp: 0.001,
    vision: 0.001,
    classification: 0.001,
    regression: 0.001,
    'time-series': 0.001,
  }
  
  let lr = baseLR[type] || 0.001
  
  // Architecture adjustments
  if (architecture === 'transformer') {
    lr = 0.0001
  }
  
  // Complexity adjustments
  if (complexity === 'complex' || complexity === 'very-complex') {
    lr *= 0.5 // Lower LR for complex models
  }
  
  // Data size adjustments
  if (context.dataSize === 'large' || context.dataSize === 'very-large') {
    lr *= 1.2 // Can use higher LR with more data
  }
  
  return Math.max(0.00001, Math.min(0.01, lr)) // Clamp between bounds
}

function determineBatchSize(complexity: string, context: Context): number {
  let batchSize = 32
  
  if (complexity === 'simple') {
    batchSize = 64
  } else if (complexity === 'complex') {
    batchSize = 16
  } else if (complexity === 'very-complex') {
    batchSize = 8
  }
  
  // Constraint adjustments
  if (context.constraints?.includes('memory')) {
    batchSize = Math.min(batchSize, 16)
  }
  
  if (context.dataSize === 'small') {
    batchSize = Math.min(batchSize, 16)
  }
  
  return batchSize
}

function determineEpochs(type: ModelSpec['type'], complexity: string, context: Context): number {
  let epochs = 10
  
  if (type === 'generative') {
    epochs = 50
  } else if (type === 'nlp' && complexity === 'complex') {
    epochs = 20
  } else if (type === 'vision') {
    epochs = 15
  }
  
  if (complexity === 'complex' || complexity === 'very-complex') {
    epochs = Math.floor(epochs * 1.5)
  }
  
  if (context.dataSize === 'small') {
    epochs = Math.floor(epochs * 1.5) // More epochs for small datasets
  }
  
  return epochs
}

function determineDropout(architecture: ModelSpec['architecture'], complexity: string, context: Context): number {
  if (context.constraints?.includes('memory')) {
    return 0.1 // Lower dropout to save memory
  }
  
  if (architecture === 'transformer') {
    return 0.1
  }
  
  if (complexity === 'simple') {
    return 0.2
  } else if (complexity === 'complex' || complexity === 'very-complex') {
    return 0.3
  }
  
  return 0.2
}

function determineActivation(architecture: ModelSpec['architecture'], type: ModelSpec['type'], context: Context): string {
  if (architecture === 'transformer') {
    return 'gelu'
  }
  
  if (context.performance === 'speed') {
    return 'relu' // Faster
  }
  
  return 'relu'
}

function determineOutputActivation(type: ModelSpec['type']): string {
  if (type === 'classification') {
    return 'softmax'
  }
  if (type === 'regression') {
    return 'linear'
  }
  if (type === 'generative') {
    return 'softmax'
  }
  return 'linear'
}

function determineMetrics(type: ModelSpec['type']): string[] {
  if (type === 'classification') {
    return ['accuracy', 'precision', 'recall', 'f1']
  }
  if (type === 'regression') {
    return ['mse', 'mae', 'r2']
  }
  if (type === 'nlp') {
    return ['accuracy', 'perplexity']
  }
  return ['accuracy']
}

function determineLoss(type: ModelSpec['type']): string {
  if (type === 'regression') {
    return 'mean_squared_error'
  }
  if (type === 'classification') {
    return 'sparse_categorical_crossentropy'
  }
  if (type === 'generative') {
    return 'sparse_categorical_crossentropy'
  }
  return 'sparse_categorical_crossentropy'
}


