/**
 * Model Status Service - Manages model status and state
 */

import { logger, performanceMonitor } from '../modules/utilities'
import type { ModelStatus } from '../core/types'

// Store for model statuses
const modelStatuses = new Map<string, ModelStatus>()

/**
 * Get model status
 */
export async function getModelStatus(modelId: string): Promise<ModelStatus> {
  const perfId = performanceMonitor.start('getModelStatus', { modelId })

  try {
    if (!modelId) {
      throw new Error('Model ID is required')
    }

    const status = modelStatuses.get(modelId)
    if (!status) {
      logger.warn('Model status not found', { modelId })
      performanceMonitor.end(perfId, { found: false })
      return {
        status: 'failed',
        error: 'Model not found',
      }
    }

    logger.debug('Model status retrieved', { modelId, status: status.status })
    performanceMonitor.end(perfId, { found: true, status: status.status })
    return status
  } catch (error) {
    logger.error(
      'Error getting model status',
      error instanceof Error ? error : new Error(String(error)),
      { modelId }
    )
    performanceMonitor.end(perfId, {
      found: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    })
    throw error
  }
}

/**
 * Set model status
 */
export function setModelStatus(
  modelId: string,
  status: Partial<ModelStatus>
): void {
  const current = modelStatuses.get(modelId) || { status: 'creating' }
  modelStatuses.set(modelId, { ...current, ...status })
  logger.debug('Model status updated', { modelId, status })
}

/**
 * Update model progress
 */
export function updateModelProgress(
  modelId: string,
  progress: number,
  currentStep?: string
): void {
  const current = modelStatuses.get(modelId)
  if (current) {
    modelStatuses.set(modelId, {
      ...current,
      progress,
      currentStep,
    })
  }
}

/**
 * Mark model as completed
 */
export function markModelCompleted(
  modelId: string,
  githubUrl?: string | null
): void {
  modelStatuses.set(modelId, {
    status: 'completed',
    progress: 100,
    currentStep: 'Modelo completado',
    githubUrl,
  })
  logger.info('Model marked as completed', { modelId })
}

/**
 * Mark model as failed
 */
export function markModelFailed(modelId: string, error: string): void {
  modelStatuses.set(modelId, {
    status: 'failed',
    error,
    progress: 0,
    currentStep: 'Error al crear el modelo',
  })
  logger.error('Model marked as failed', { modelId, error })
}

/**
 * Get all model statuses
 */
export function getAllModelStatuses(): Map<string, ModelStatus> {
  return modelStatuses
}

/**
 * Clear model status
 */
export function clearModelStatus(modelId: string): void {
  modelStatuses.delete(modelId)
}


