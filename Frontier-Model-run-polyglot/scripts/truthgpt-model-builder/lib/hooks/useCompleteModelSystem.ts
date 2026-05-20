/**
 * Hook maestro que integra TODO el sistema de modelos
 * ====================================================
 * 
 * Una solución completa lista para usar
 */

import { useCallback } from 'react'
import { useIntegratedModelCreation } from './useIntegratedModelCreation'
import { useModelAnalytics } from './useModelAnalytics'
import { useModelOptimizer } from './useModelOptimizer'
import { useModelValidator } from './useModelValidator'
import { useModelHistory } from './useModelHistory'
import { useModelTemplates } from './useModelTemplates'
import { useModelComparison } from './useModelComparison'
import { useModelNotifications } from './useModelNotifications'
import { TruthGPTAPIClient } from '../truthgpt-api-client'
import { generateUniqueModelName, extractTags } from '../modelUtils'

export interface CompleteModelSystemOptions {
  enableAnalytics?: boolean
  enableOptimization?: boolean
  enableValidation?: boolean
  enableHistory?: boolean
  enableTemplates?: boolean
  enableComparison?: boolean
  enableNotifications?: boolean
}

export interface CreateModelWithSystemOptions {
  description: string
  spec?: any
  templateId?: string
  useTemplate?: boolean
  enableOptimization?: boolean
  enableValidation?: boolean
  tags?: string[]
  notes?: string
}

export interface UseCompleteModelSystemResult {
  // Creación
  createModel: (options: CreateModelWithSystemOptions) => Promise<string | null>
  validateDescription: (description: string) => void
  
  // Optimización
  getOptimizationSuggestions: (description: string, spec?: any) => any
  getComplexityEstimate: (description: string) => any
  
  // Historial
  history: ReturnType<typeof useModelHistory>['history']
  searchHistory: (query: string) => any[]
  getHistoryStats: () => any
  
  // Plantillas
  templates: ReturnType<typeof useModelTemplates>['templates']
  createFromTemplate: (templateId: string, customizations?: any) => any
  searchTemplates: (query: string) => any[]
  
  // Comparación
  createComparison: (modelIds: string[], metrics: any) => string
  getBestModel: (comparisonId: string, metric: string) => string | null
  
  // Analytics
  analytics: ReturnType<typeof useModelAnalytics>['analytics']
  getAnalyticsStats: () => any
  exportAnalytics: () => string
  
  // Estado
  isCreating: boolean
  isValidationPending: boolean
  validationResult: any
  activeModels: Set<string>
  
  // Utilidades
  clearAll: () => void
  exportAll: () => string
}

/**
 * Hook maestro que integra todo el sistema
 */
export function useCompleteModelSystem(
  apiClient: TruthGPTAPIClient | null,
  apiConnected: boolean,
  options: CompleteModelSystemOptions = {}
): UseCompleteModelSystemResult {
  const {
    enableAnalytics = true,
    enableOptimization = true,
    enableValidation = true,
    enableHistory = true,
    enableTemplates = true,
    enableComparison = true,
    enableNotifications = true
  } = options

  // Hooks principales
  const integratedCreation = useIntegratedModelCreation(apiClient, apiConnected)
  const analytics = enableAnalytics ? useModelAnalytics() : null
  const optimizer = enableOptimization ? useModelOptimizer() : null
  const validator = enableValidation ? useModelValidator() : null
  const history = enableHistory ? useModelHistory({ enablePersistence: true }) : null
  const templates = enableTemplates ? useModelTemplates() : null
  const comparison = enableComparison ? useModelComparison() : null
  const notifications = enableNotifications ? useModelNotifications() : null

  // Crear modelo con todo el sistema
  const createModel = useCallback(
    async (options: CreateModelWithSystemOptions): Promise<string | null> => {
      const {
        description,
        spec,
        templateId,
        useTemplate = false,
        enableOptimization = true,
        enableValidation = true,
        tags,
        notes
      } = options

      let finalSpec = spec
      let finalDescription = description

      // Usar plantilla si se especifica
      if (useTemplate && templateId && templates) {
        try {
          const templateSpec = templates.createFromTemplate(templateId, spec)
          finalSpec = templateSpec
          notifications?.notifyInfo(`Usando plantilla: ${templates.getTemplate(templateId)?.name}`)
        } catch (error) {
          notifications?.notifyWarning(`Error al usar plantilla: ${error}`)
        }
      }

      // Optimización
      if (enableOptimization && optimizer) {
        const suggestions = optimizer.optimizeModel(finalDescription, finalSpec)
        
        if (suggestions.optimizer && !finalSpec?.optimizer) {
          finalSpec = { ...finalSpec, optimizer: suggestions.optimizer.suggestion }
        }
        
        if (suggestions.loss && !finalSpec?.loss) {
          finalSpec = { ...finalSpec, loss: suggestions.loss.suggestion }
        }

        if (suggestions.warnings && suggestions.warnings.length > 0) {
          suggestions.warnings.forEach(warning => {
            notifications?.notifyWarning(warning)
          })
        }
      }

      // Validación
      if (enableValidation && validator) {
        const validation = validator.validateComplete(finalDescription, finalSpec)
        
        if (!validation.valid) {
          validation.errors.forEach(error => {
            notifications?.notifyError('', new Error(error))
          })
          return null
        }

        if (validation.warnings.length > 0) {
          validation.warnings.forEach(warning => {
            notifications?.notifyWarning(warning)
          })
        }
      }

      // Extraer tags automáticamente si no se proporcionan
      const finalTags = tags || extractTags(finalDescription)

      // Registrar inicio en analytics
      const modelId = `model-${Date.now()}`
      analytics?.recordCreation(modelId)

      // Agregar al historial
      const historyId = history?.addModel({
        name: finalSpec?.modelName || generateUniqueModelName('Model', []),
        description: finalDescription,
        spec: finalSpec,
        status: 'creating',
        tags: finalTags,
        notes
      })

      // Crear modelo
      const createdModelId = await integratedCreation.createModel({
        description: finalDescription,
        spec: finalSpec,
        enableCache: true,
        enableValidation: enableValidation,
        enableAnalysis: true
      })

      if (createdModelId) {
        // Actualizar historial
        if (historyId && history) {
          history.updateModel(historyId, {
            id: createdModelId,
            status: 'creating'
          })
        }

        notifications?.notifyModelCreated(createdModelId, finalSpec?.modelName || 'Model')
        
        return createdModelId
      }

      return null
    },
    [integratedCreation, analytics, optimizer, validator, history, templates, notifications]
  )

  const clearAll = useCallback(() => {
    integratedCreation.clearCache()
    integratedCreation.clearQueue()
    analytics?.reset()
    history?.clear()
    comparison?.clear()
  }, [integratedCreation, analytics, history, comparison])

  const exportAll = useCallback(() => {
    const data = {
      analytics: analytics ? analytics.exportData() : null,
      history: history ? history.exportHistory() : null,
      exportedAt: new Date().toISOString()
    }
    return JSON.stringify(data, null, 2)
  }, [analytics, history])

  return {
    // Creación
    createModel,
    validateDescription: integratedCreation.validateDescription,
    
    // Optimización
    getOptimizationSuggestions: optimizer?.optimizeModel || (() => ({})),
    getComplexityEstimate: optimizer?.getComplexityEstimate || (() => ({ complexity: 'medium' })),
    
    // Historial
    history: history?.history || [],
    searchHistory: history?.searchModels || (() => []),
    getHistoryStats: history?.getStats || (() => ({ total: 0 })),
    
    // Plantillas
    templates: templates?.templates || [],
    createFromTemplate: templates?.createFromTemplate || (() => ({})),
    searchTemplates: templates?.searchTemplates || (() => []),
    
    // Comparación
    createComparison: comparison?.createComparison || (() => ''),
    getBestModel: comparison?.getBestModel || (() => null),
    
    // Analytics
    analytics: analytics?.analytics || {
      totalCreated: 0,
      totalCompleted: 0,
      totalFailed: 0,
      averageCreationTime: 0,
      successRate: 0,
      errorRate: 0,
      mostCommonErrors: new Map(),
      timeSeries: []
    },
    getAnalyticsStats: analytics?.getStats || (() => ({})),
    exportAnalytics: analytics?.exportData || (() => ''),
    
    // Estado
    isCreating: integratedCreation.isCreating,
    isValidationPending: integratedCreation.isValidationPending,
    validationResult: integratedCreation.validationResult,
    activeModels: integratedCreation.activeModels,
    
    // Utilidades
    clearAll,
    exportAll
  }
}

