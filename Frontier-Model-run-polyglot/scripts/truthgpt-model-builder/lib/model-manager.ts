/**
 * Model management utilities
 */

import { logger } from './logger'
import { performanceMonitor } from './performance'
import { getModelStatus } from './truthgpt-service'
import type { ModelStatus } from './core/types'

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

export class ModelManager {
  private static models = new Map<string, ModelInfo>()

  /**
   * List all models
   */
  static async listModels(limit?: number, offset?: number): Promise<ModelInfo[]> {
    const perfId = performanceMonitor.start('listModels', { limit, offset })

    try {
      const models = Array.from(this.models.values())
        .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())

      let result = models
      if (offset !== undefined) {
        result = result.slice(offset)
      }
      if (limit !== undefined) {
        result = result.slice(0, limit)
      }

      logger.debug('Models listed', { count: result.length, total: models.length })
      performanceMonitor.end(perfId, { count: result.length })
      return result
    } catch (error) {
      logger.error('Error listing models', error instanceof Error ? error : new Error(String(error)))
      performanceMonitor.end(perfId, { error: error instanceof Error ? error.message : 'Unknown error' })
      throw error
    }
  }

  /**
   * Get model by ID
   */
  static async getModel(modelId: string): Promise<ModelInfo | null> {
    const perfId = performanceMonitor.start('getModel', { modelId })

    try {
      const model = this.models.get(modelId)
      if (!model) {
        logger.warn('Model not found', { modelId })
        performanceMonitor.end(perfId, { found: false })
        return null
      }

      // Update status from service
      const status = await getModelStatus(modelId)
      if (model.status !== status.status) {
        model.status = status.status
        model.updatedAt = new Date()
        if (status.progress !== undefined) {
          model.progress = status.progress
        }
        if (status.spec) {
          model.spec = status.spec
        }
      }

      logger.debug('Model retrieved', { modelId, status: model.status })
      performanceMonitor.end(perfId, { found: true })
      return model
    } catch (error) {
      logger.error('Error getting model', error instanceof Error ? error : new Error(String(error)), { modelId })
      performanceMonitor.end(perfId, { found: false, error: error instanceof Error ? error.message : 'Unknown error' })
      throw error
    }
  }

  /**
   * Register a new model
   */
  static registerModel(modelInfo: ModelInfo): void {
    this.models.set(modelInfo.id, {
      ...modelInfo,
      createdAt: modelInfo.createdAt || new Date(),
      updatedAt: new Date(),
    })
    logger.info('Model registered', { modelId: modelInfo.id, name: modelInfo.name })
  }

  /**
   * Update model information
   */
  static updateModel(modelId: string, updates: Partial<ModelInfo>): boolean {
    const model = this.models.get(modelId)
    if (!model) {
      logger.warn('Model not found for update', { modelId })
      return false
    }

    this.models.set(modelId, {
      ...model,
      ...updates,
      updatedAt: new Date(),
    })

    logger.debug('Model updated', { modelId, updates: Object.keys(updates) })
    return true
  }

  /**
   * Delete model
   */
  static deleteModel(modelId: string): boolean {
    const deleted = this.models.delete(modelId)
    if (deleted) {
      logger.info('Model deleted', { modelId })
    } else {
      logger.warn('Model not found for deletion', { modelId })
    }
    return deleted
  }

  /**
   * Get models by status
   */
  static getModelsByStatus(status: ModelStatus['status']): ModelInfo[] {
    return Array.from(this.models.values()).filter(m => m.status === status)
  }

  /**
   * Get model statistics
   */
  static getStatistics(): {
    total: number
    byStatus: Record<string, number>
    oldest: Date | null
    newest: Date | null
  } {
    const models = Array.from(this.models.values())
    const byStatus: Record<string, number> = {}
    let oldest: Date | null = null
    let newest: Date | null = null

    for (const model of models) {
      byStatus[model.status] = (byStatus[model.status] || 0) + 1

      if (!oldest || model.createdAt < oldest) {
        oldest = model.createdAt
      }
      if (!newest || model.createdAt > newest) {
        newest = model.createdAt
      }
    }

    return {
      total: models.length,
      byStatus,
      oldest,
      newest,
    }
  }

  /**
   * Clear all models (for testing)
   */
  static clear(): void {
    this.models.clear()
    logger.info('All models cleared')
  }
}


