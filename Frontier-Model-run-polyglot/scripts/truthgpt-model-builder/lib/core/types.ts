/**
 * Core types and interfaces for the TruthGPT Model Builder
 */

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

export interface ModelStatus {
  status: 'creating' | 'completed' | 'failed'
  githubUrl?: string | null
  error?: string
  progress?: number
  currentStep?: string
  spec?: ModelSpec
}

export interface ModelInfo {
  id: string
  name: string
  description: string
  status: ModelStatus['status']
  createdAt: Date
  updatedAt: Date
  progress?: number
  spec?: any
}


