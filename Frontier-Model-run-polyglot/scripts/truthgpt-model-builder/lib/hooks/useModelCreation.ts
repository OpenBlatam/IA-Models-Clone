/**
 * Custom hook for model creation with enhanced features
 */

import { useState, useCallback } from 'react'
import { createTruthGPTModel, getModelStatus } from '../truthgpt-service'
import { generateModelId, sanitizeModelName } from '../modules/utilities'
import { analyzeModelDescription } from '../modules/analysis'
import { validateModelSpec } from '../modules/validation'
import { optimizeModelSpec } from '../modules/optimization'
import { adaptToTruthGPT } from '../modules/adaptation'
import { estimateCost } from '../modules/optimization'
import { logger } from '../modules/utilities'
import type { ModelSpec } from '../core/types'

interface UseModelCreationOptions {
  onSuccess?: (modelId: string) => void
  onError?: (error: Error) => void
  onProgress?: (progress: number, step: string) => void
}

export function useModelCreation(options: UseModelCreationOptions = {}) {
  const [isCreating, setIsCreating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState<string>('')
  const [error, setError] = useState<Error | null>(null)
  const [previewSpec, setPreviewSpec] = useState<ModelSpec | null>(null)
  const [costEstimate, setCostEstimate] = useState<any>(null)

  const analyzeDescription = useCallback(async (description: string) => {
    try {
      // Analyze
      const spec = analyzeModelDescription(description)
      
      // Validate
      const validation = validateModelSpec(spec)
      if (!validation.isValid && validation.errors.length > 0) {
        throw new Error(validation.errors[0])
      }

      // Optimize
      const optimization = optimizeModelSpec(spec)
      
      // Adapt to TruthGPT
      const truthgptSpec = adaptToTruthGPT(optimization.optimized)
      
      // Estimate cost
      const cost = estimateCost(truthgptSpec)

      setPreviewSpec(truthgptSpec)
      setCostEstimate(cost)

      return {
        spec: truthgptSpec,
        optimization,
        validation,
        cost,
      }
    } catch (err) {
      logger.error('Error analyzing description', err as Error)
      throw err
    }
  }, [])

  const createModel = useCallback(async (
    description: string,
    modelName?: string
  ) => {
    setIsCreating(true)
    setError(null)
    setProgress(0)

    try {
      // Generate model ID and name
      const modelId = generateModelId()
      const finalModelName = modelName || sanitizeModelName(description.slice(0, 50))

      // Start creation
      createTruthGPTModel(modelId, finalModelName, description)

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getModelStatus(modelId)
          
          if (status.progress !== undefined) {
            setProgress(status.progress)
            options.onProgress?.(status.progress, status.currentStep || '')
          }

          if (status.currentStep) {
            setCurrentStep(status.currentStep)
          }

          if (status.status === 'completed') {
            clearInterval(pollInterval)
            setIsCreating(false)
            setProgress(100)
            options.onSuccess?.(modelId)
          } else if (status.status === 'failed') {
            clearInterval(pollInterval)
            setIsCreating(false)
            const err = new Error(status.error || 'Model creation failed')
            setError(err)
            options.onError?.(err)
          }
        } catch (err) {
          logger.error('Error polling model status', err as Error)
        }
      }, 1000) // Poll every second

      // Cleanup on unmount
      return () => clearInterval(pollInterval)
    } catch (err) {
      setIsCreating(false)
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      options.onError?.(error)
      throw error
    }
  }, [options])

  const reset = useCallback(() => {
    setIsCreating(false)
    setProgress(0)
    setCurrentStep('')
    setError(null)
    setPreviewSpec(null)
    setCostEstimate(null)
  }, [])

  return {
    isCreating,
    progress,
    currentStep,
    error,
    previewSpec,
    costEstimate,
    analyzeDescription,
    createModel,
    reset,
  }
}


