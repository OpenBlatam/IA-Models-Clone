/**
 * Model Analyzer - Analyzes user descriptions to determine optimal model architecture
 */

import { toPascalCase, toPascalCaseLoss, capitalizeFirst, formatMetricsForCode } from '../../utils/code-generators'

export interface ModelSpec {
  type: 'classification' | 'regression' | 'nlp' | 'vision' | 'time-series' | 'generative' | 'custom'
  architecture: 'dense' | 'cnn' | 'rnn' | 'lstm' | 'transformer' | 'hybrid'
  layers: number[]
  activation: string
  outputActivation: string
  useDropout: boolean
  dropoutRate: number
  useBatchNorm: boolean
  optimizer: 'adam' | 'sgd' | 'rmsprop' | 'adamw'
  learningRate: number
  batchSize: number
  epochs: number
  metrics: string[]
  loss: string
}

const keywords = {
  classification: ['clasificar', 'classify', 'clasificación', 'categoría', 'category', 'etiqueta', 'label'],
  regression: ['predecir', 'predict', 'regresión', 'regression', 'valor', 'value', 'numérico', 'numeric'],
  nlp: ['texto', 'text', 'nlp', 'lenguaje', 'language', 'sentimiento', 'sentiment', 'análisis', 'analysis', 'traducción', 'translation'],
  vision: ['imagen', 'image', 'imágenes', 'images', 'visual', 'reconocimiento', 'recognition', 'detección', 'detection', 'cámara', 'camera'],
  'time-series': ['tiempo', 'time', 'serie', 'series', 'temporal', 'temporal', 'secuencia', 'sequence', 'predicción', 'forecast'],
  generative: ['generar', 'generate', 'crear', 'create', 'generativo', 'generative', 'gpt', 'llm', 'lenguaje', 'language model']
}

import { adaptiveAnalyze } from '../../adaptive-analyzer'
import { enhancedAnalyze } from '../../enhanced-analyzer'

export function analyzeModelDescription(description: string): ModelSpec {
  // Use enhanced analyzer for ultra-intelligent results
  return enhancedAnalyze(description)
}

// Keep original function for backward compatibility
export function analyzeModelDescriptionLegacy(description: string): ModelSpec {
  const lowerDesc = description.toLowerCase()
  
  // Determine model type
  let type: ModelSpec['type'] = 'custom'
  let maxMatches = 0
  
  for (const [modelType, words] of Object.entries(keywords)) {
    const matches = words.filter(word => lowerDesc.includes(word)).length
    if (matches > maxMatches) {
      maxMatches = matches
      type = modelType as ModelSpec['type']
    }
  }
  
  // Determine architecture
  let architecture: ModelSpec['architecture'] = 'dense'
  
  if (type === 'vision') {
    architecture = 'cnn'
  } else if (type === 'nlp' || type === 'generative') {
    architecture = lowerDesc.includes('transformer') || lowerDesc.includes('gpt') || lowerDesc.includes('llm')
      ? 'transformer'
      : lowerDesc.includes('lstm') || lowerDesc.includes('rnn')
      ? 'lstm'
      : 'transformer'
  } else if (type === 'time-series') {
    architecture = 'lstm'
  } else if (lowerDesc.includes('cnn') || lowerDesc.includes('convolutional')) {
    architecture = 'cnn'
  } else if (lowerDesc.includes('lstm') || lowerDesc.includes('rnn')) {
    architecture = 'lstm'
  } else if (lowerDesc.includes('transformer')) {
    architecture = 'transformer'
  } else if (lowerDesc.includes('híbrido') || lowerDesc.includes('hybrid')) {
    architecture = 'hybrid'
  }
  
  // Determine layer sizes based on type and description
  let layers: number[] = []
  if (architecture === 'cnn') {
    layers = [64, 128, 256, 128, 64]
  } else if (architecture === 'lstm') {
    layers = [128, 64, 32]
  } else if (architecture === 'transformer') {
    layers = [512, 256, 128]
  } else {
    // Dense layers
    if (lowerDesc.includes('simple') || lowerDesc.includes('simple')) {
      layers = [64, 32]
    } else if (lowerDesc.includes('complex') || lowerDesc.includes('complejo')) {
      layers = [512, 256, 128, 64]
    } else {
      layers = [256, 128, 64]
    }
  }
  
  // Determine activation functions
  const activation = lowerDesc.includes('relu') ? 'relu' 
    : lowerDesc.includes('tanh') ? 'tanh'
    : lowerDesc.includes('gelu') ? 'gelu'
    : 'relu'
  
  // Determine output activation
  let outputActivation = 'linear'
  if (type === 'classification') {
    const numClasses = extractNumber(lowerDesc, 'clases', 'classes') || 10
    outputActivation = numClasses === 2 ? 'sigmoid' : 'softmax'
  } else if (type === 'generative') {
    outputActivation = 'softmax'
  }
  
  // Determine optimizer
  let optimizer: ModelSpec['optimizer'] = 'adam'
  if (lowerDesc.includes('sgd')) {
    optimizer = 'sgd'
  } else if (lowerDesc.includes('rmsprop')) {
    optimizer = 'rmsprop'
  } else if (lowerDesc.includes('adamw')) {
    optimizer = 'adamw'
  }
  
  // Determine learning rate
  const learningRate = extractNumber(lowerDesc, 'learning rate', 'tasa de aprendizaje') || 
    (type === 'generative' ? 0.0001 : 0.001)
  
  // Determine loss function
  let loss = 'sparse_categorical_crossentropy'
  if (type === 'regression') {
    loss = 'mean_squared_error'
  } else if (type === 'classification' && extractNumber(lowerDesc, 'clases', 'classes') === 2) {
    loss = 'binary_crossentropy'
  }
  
  // Determine metrics
  const metrics = type === 'classification' 
    ? ['accuracy', 'precision', 'recall']
    : type === 'regression'
    ? ['mse', 'mae', 'r2']
    : ['accuracy']
  
  return {
    type,
    architecture,
    layers,
    activation,
    outputActivation,
    useDropout: true,
    dropoutRate: architecture === 'transformer' ? 0.1 : 0.2,
    useBatchNorm: architecture === 'cnn',
    optimizer,
    learningRate,
    batchSize: 32,
    epochs: type === 'generative' ? 50 : 10,
    metrics,
    loss
  }
}

function extractNumber(text: string, ...keywords: string[]): number | null {
  for (const keyword of keywords) {
    const regex = new RegExp(`${keyword}\\s*(?:=|:|de|of)?\\s*(\\d+)`, 'i')
    const match = text.match(regex)
    if (match) {
      return parseInt(match[1], 10)
    }
  }
  return null
}

export function generateArchitectureCode(spec: ModelSpec, modelName: string, description: string): string {
  const className = toPascalCase(modelName)
  
  if (spec.architecture === 'cnn') {
    return generateCNNCode(spec, className, modelName, description)
  } else if (spec.architecture === 'lstm') {
    return generateLSTMCode(spec, className, modelName, description)
  } else if (spec.architecture === 'transformer') {
    return generateTransformerCode(spec, className, modelName, description)
  } else {
    return generateDenseCode(spec, className, modelName, description)
  }
}

function generateDenseCode(spec: ModelSpec, className: string, modelName: string, description: string): string {
  const layers = spec.layers.map((size, idx) => {
    const isLast = idx === spec.layers.length - 1
    const activation = isLast ? spec.outputActivation : spec.activation
    const dropout = !isLast && spec.useDropout ? `tg.layers.Dropout(${spec.dropoutRate}),` : ''
    const batchNorm = !isLast && spec.useBatchNorm ? `tg.layers.BatchNormalization(),` : ''
    
    return `            tg.layers.Dense(${size}, activation='${activation}'),${dropout ? '\n' + '            ' + dropout : ''}${batchNorm ? '\n' + '            ' + batchNorm : ''}`
  }).join(',\n')
  
  return `"""
${className} - TruthGPT Model
Generated based on: ${description}
Type: ${spec.type}
Architecture: ${spec.architecture}
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import torch
import numpy as np


class ${className}(tg.Sequential):
    """
    TruthGPT Model: ${description}
    Architecture: ${spec.architecture}
    """
    
    def __init__(self, input_size=${spec.layers[0] * 2}, output_size=10, **kwargs):
        """
        Initialize the ${className} model.
        """
        layers = [
${layers}
        ]
        
        super().__init__(layers, name='${modelName}', **kwargs)
    
    def compile(self, **kwargs):
        """Compile the model with TruthGPT optimizations."""
        optimizer_name = '${capitalizeFirst(spec.optimizer)}'
        learning_rate = ${spec.learningRate}
        if optimizer_name == 'Adam':
            optimizer = kwargs.get('optimizer', tg.optimizers.Adam(learning_rate=learning_rate))
        elif optimizer_name == 'Sgd':
            optimizer = kwargs.get('optimizer', tg.optimizers.SGD(learning_rate=learning_rate))
        elif optimizer_name == 'Rmsprop':
            optimizer = kwargs.get('optimizer', tg.optimizers.RMSprop(learning_rate=learning_rate))
        elif optimizer_name == 'Adamw':
            optimizer = kwargs.get('optimizer', tg.optimizers.AdamW(learning_rate=learning_rate))
        else:
            optimizer = kwargs.get('optimizer', tg.optimizers.Adam(learning_rate=learning_rate))
        
        loss_name = '${toPascalCaseLoss(spec.loss)}'
        if loss_name == 'SparseCategoricalCrossentropy':
            loss = kwargs.get('loss', tg.losses.SparseCategoricalCrossentropy())
        elif loss_name == 'MeanSquaredError':
            loss = kwargs.get('loss', tg.losses.MeanSquaredError())
        elif loss_name == 'BinaryCrossentropy':
            loss = kwargs.get('loss', tg.losses.BinaryCrossentropy())
        else:
            loss = kwargs.get('loss', tg.losses.SparseCategoricalCrossentropy())
        
        metrics = kwargs.get('metrics', ${formatMetricsForCode(spec.metrics)})
        
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, x, y, epochs=${spec.epochs}, batch_size=${spec.batchSize}, **kwargs):
        """Train the model with TruthGPT optimizations."""
        return super().fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs)
`
}

function generateCNNCode(spec: ModelSpec, className: string, modelName: string, description: string): string {
  return `"""
${className} - TruthGPT CNN Model
Generated based: ${description}
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import torch
import numpy as np


class ${className}(tg.Sequential):
    """
    TruthGPT CNN Model: ${description}
    """
    
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, **kwargs):
        layers = [
            tg.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tg.layers.MaxPooling2D(2),
            tg.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tg.layers.MaxPooling2D(2),
            tg.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tg.layers.MaxPooling2D(2),
            tg.layers.Flatten(),
            tg.layers.Dense(128, activation='relu'),
            tg.layers.Dropout(0.5),
            tg.layers.Dense(num_classes, activation='softmax')
        ]
        
        super().__init__(layers, name='${modelName}', **kwargs)
    
    def compile(self, **kwargs):
        optimizer = kwargs.get('optimizer', tg.optimizers.Adam(learning_rate=0.001))
        loss = kwargs.get('loss', tg.losses.SparseCategoricalCrossentropy())
        metrics = kwargs.get('metrics', ['accuracy'])
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
`
}

function generateLSTMCode(spec: ModelSpec, className: string, modelName: string, description: string): string {
  return `"""
${className} - TruthGPT LSTM Model
Generated based: ${description}
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import torch
import numpy as np


class ${className}(tg.Sequential):
    """
    TruthGPT LSTM Model: ${description}
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, num_classes=10, **kwargs):
        layers = [
            tg.layers.Embedding(vocab_size, embedding_dim),
            tg.layers.LSTM(${spec.layers[0]}, return_sequences=True),
            tg.layers.Dropout(${spec.dropoutRate}),
            tg.layers.LSTM(${spec.layers[1] || 64}, return_sequences=False),
            tg.layers.Dropout(${spec.dropoutRate}),
            tg.layers.Dense(${spec.layers[2] || 32}, activation='relu'),
            tg.layers.Dense(num_classes, activation='softmax')
        ]
        
        super().__init__(layers, name='${modelName}', **kwargs)
    
    def compile(self, **kwargs):
        optimizer = kwargs.get('optimizer', tg.optimizers.Adam(learning_rate=0.001))
        loss = kwargs.get('loss', tg.losses.SparseCategoricalCrossentropy())
        metrics = kwargs.get('metrics', ['accuracy'])
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
`
}

function generateTransformerCode(spec: ModelSpec, className: string, modelName: string, description: string): string {
  return `"""
${className} - TruthGPT Transformer Model
Generated based: ${description}
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import truthgpt as tg
import torch
import numpy as np


class ${className}(tg.Sequential):
    """
    TruthGPT Transformer Model: ${description}
    """
    
    def __init__(self, vocab_size=50000, embedding_dim=512, num_heads=8, num_layers=6, num_classes=10, **kwargs):
        layers = [
            tg.layers.Embedding(vocab_size, embedding_dim),
            tg.layers.TransformerBlock(embedding_dim, num_heads, num_layers),
            tg.layers.GlobalAveragePooling1D(),
            tg.layers.Dense(${spec.layers[0]}, activation='gelu'),
            tg.layers.Dropout(${spec.dropoutRate}),
            tg.layers.Dense(num_classes, activation='softmax')
        ]
        
        super().__init__(layers, name='${modelName}', **kwargs)
    
    def compile(self, **kwargs):
        optimizer = kwargs.get('optimizer', tg.optimizers.AdamW(learning_rate=0.0001))
        loss = kwargs.get('loss', tg.losses.SparseCategoricalCrossentropy())
        metrics = kwargs.get('metrics', ['accuracy'])
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
`
}

// Utility functions imported from utils/code-generators

