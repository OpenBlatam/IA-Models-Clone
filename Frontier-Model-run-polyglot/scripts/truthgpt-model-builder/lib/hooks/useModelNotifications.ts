/**
 * Hook para notificaciones inteligentes de modelos
 * =================================================
 */

import { useCallback } from 'react'
import { toast } from 'react-hot-toast'

export interface ModelNotificationOptions {
  showProgress?: boolean
  showSuccess?: boolean
  showErrors?: boolean
  showWarnings?: boolean
  progressInterval?: number // Porcentaje para mostrar progreso
}

/**
 * Hook para notificaciones inteligentes durante la creación de modelos
 */
export function useModelNotifications(options: ModelNotificationOptions = {}) {
  const {
    showProgress = true,
    showSuccess = true,
    showErrors = true,
    showWarnings = true,
    progressInterval = 25
  } = options

  const notifyModelCreated = useCallback((modelId: string, modelName: string) => {
    if (showSuccess) {
      toast.success(`Modelo ${modelName} creado`, {
        icon: '✅',
        duration: 3000,
        id: `model-created-${modelId}`
      })
    }
  }, [showSuccess])

  const notifyProgress = useCallback((modelId: string, progress: number, step?: string) => {
    if (showProgress && progress % progressInterval === 0) {
      const message = step ? `${step} (${progress}%)` : `Progreso: ${progress}%`
      toast(message, {
        icon: '⏳',
        duration: 2000,
        id: `model-progress-${modelId}`
      })
    }
  }, [showProgress, progressInterval])

  const notifyModelCompleted = useCallback((modelId: string, githubUrl?: string) => {
    if (showSuccess) {
      const message = githubUrl 
        ? `¡Modelo completado! ${githubUrl}`
        : '¡Modelo completado!'
      
      toast.success(message, {
        icon: '🎉',
        duration: 5000,
        id: `model-completed-${modelId}`
      })
    }
  }, [showSuccess])

  const notifyError = useCallback((modelId: string, error: Error | string) => {
    if (showErrors) {
      const message = error instanceof Error ? error.message : error
      toast.error(`Error: ${message}`, {
        icon: '❌',
        duration: 5000,
        id: `model-error-${modelId}`
      })
    }
  }, [showErrors])

  const notifyWarning = useCallback((message: string) => {
    if (showWarnings) {
      toast(message, {
        icon: '⚠️',
        duration: 3000
      })
    }
  }, [showWarnings])

  const notifyInfo = useCallback((message: string) => {
    toast(message, {
      icon: 'ℹ️',
      duration: 3000
    })
  }, [])

  return {
    notifyModelCreated,
    notifyProgress,
    notifyModelCompleted,
    notifyError,
    notifyWarning,
    notifyInfo
  }
}

