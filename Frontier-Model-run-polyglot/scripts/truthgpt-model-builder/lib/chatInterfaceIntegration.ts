/**
 * Utilidades para integrar el sistema completo en ChatInterface
 * =============================================================
 */

import { useCompleteModelSystem, useTruthGPTAPI } from './hooks'
import { useModelStore } from '@/store/modelStore'
import { saveModelToHistory } from '@/lib/storage'

/**
 * Hook helper para integrar el sistema completo en ChatInterface
 */
export function useChatInterfaceIntegration() {
  const { client, isConnected } = useTruthGPTAPI()
  const { addMessage, setCurrentModel } = useModelStore()

  const modelSystem = useCompleteModelSystem(client, isConnected, {
    enableAnalytics: true,
    enableOptimization: true,
    enableValidation: true,
    enableHistory: true,
    enableTemplates: true,
    enableComparison: true,
    enableNotifications: true
  })

  /**
   * Crea un modelo desde ChatInterface con todas las integraciones
   */
  const createModelFromChat = async (
    description: string,
    spec?: any,
    options?: {
      useTemplate?: boolean
      templateId?: string
      showMessages?: boolean
    }
  ) => {
    // Agregar mensaje del usuario
    if (options?.showMessages !== false) {
      addMessage({
        id: Date.now().toString(),
        role: 'user',
        content: description,
        timestamp: new Date()
      })
    }

    // Validar descripción
    modelSystem.validateDescription(description)

    // Crear modelo con el sistema completo
    const modelId = await modelSystem.createModel({
      description,
      spec,
      templateId: options?.templateId,
      useTemplate: options?.useTemplate || false,
      enableOptimization: true,
      enableValidation: true
    })

    if (modelId) {
      // Obtener información del modelo del historial
      const modelInfo = modelSystem.history.find(m => m.id === modelId)
      
      if (modelInfo) {
        // Actualizar modelo actual
        setCurrentModel({
          id: modelId,
          name: modelInfo.name,
          description: modelInfo.description,
          status: 'creating' as const,
          createdAt: new Date(modelInfo.createdAt),
          progress: 0,
          currentStep: 'Iniciando...',
          spec: modelInfo.spec || null
        })

        // Guardar en historial persistente
        try {
          saveModelToHistory({
            id: modelId,
            name: modelInfo.name,
            description: modelInfo.description,
            status: 'creating',
            createdAt: new Date(modelInfo.createdAt),
            spec: modelInfo.spec || null
          } as any)
        } catch (error) {
          console.error('Error saving to history:', error)
        }

        // Agregar mensaje del asistente
        if (options?.showMessages !== false) {
          addMessage({
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: `¡Perfecto! He comenzado a crear tu modelo "${modelInfo.name}" basado en: "${description}". El modelo se está generando y pronto estará disponible.`,
            timestamp: new Date()
          })
        }
      }

      return modelId
    }

    return null
  }

  /**
   * Actualiza el estado del modelo cuando se completa
   */
  const handleModelCompleted = (modelId: string, githubUrl?: string) => {
    const modelInfo = modelSystem.history.find(m => m.id === modelId)
    
    if (modelInfo) {
      // Actualizar modelo actual
      setCurrentModel({
        id: modelId,
        name: modelInfo.name,
        description: modelInfo.description,
        status: 'completed' as const,
        githubUrl: githubUrl || null,
        createdAt: new Date(modelInfo.createdAt),
        progress: 100,
        currentStep: 'Completado',
        spec: modelInfo.spec || null
      })

      // Actualizar historial
      modelSystem.history.forEach((item, index) => {
        if (item.id === modelId) {
          modelSystem.history[index] = {
            ...item,
            status: 'completed',
            completedAt: Date.now(),
            duration: item.completedAt ? item.completedAt - item.createdAt : undefined,
            githubUrl
          }
        }
      })

      // Agregar mensaje de completado
      addMessage({
        id: Date.now().toString(),
        role: 'assistant',
        content: `¡Listo! Tu modelo "${modelInfo.name}" ha sido creado exitosamente.${githubUrl ? ` Disponible en: ${githubUrl}` : ''}`,
        timestamp: new Date()
      })
    }
  }

  return {
    ...modelSystem,
    createModelFromChat,
    handleModelCompleted,
    isConnected
  }
}










