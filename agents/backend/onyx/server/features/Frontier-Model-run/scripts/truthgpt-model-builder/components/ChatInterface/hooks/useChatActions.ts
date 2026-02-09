import { useCallback } from 'react'
import { useModelStore } from '@/store/modelStore'
import { analyzeModelDescription } from '@/lib/modules/management'
import { validateUserDescription as validateDescription } from '@/lib/modules/validation'

export function useChatActions() {
  const { addModel, updateModel } = useModelStore()

  const handleSendMessage = useCallback(
    async (description: string) => {
      try {
        const validation = validateDescription(description)
        if (!validation.valid) {
          return { success: false, error: validation.errors.join(', ') }
        }

        const analysis = await analyzeModelDescription(description)
        
        const modelId = `model-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        const modelName = `truthgpt-${analysis.type.toLowerCase().replace(/\s+/g, '-')}`

        const newModel = {
          id: modelId,
          name: modelName,
          description,
          status: 'creating' as const,
          spec: analysis,
          createdAt: new Date().toISOString(),
        }

        addModel(newModel)

        const response = await fetch('/api/create-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ description }),
        })

        if (!response.ok) {
          throw new Error('Failed to create model')
        }

        const data = await response.json()
        updateModel(modelId, { status: data.status, githubUrl: data.githubUrl })

        return { success: true, modelId, model: newModel }
      } catch (error) {
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        }
      }
    },
    [addModel, updateModel]
  )

  const handleValidateDescription = useCallback((description: string) => {
    return validateDescription(description)
  }, [])

  const handlePreviewModel = useCallback(async (description: string) => {
    try {
      const analysis = await analyzeModelDescription(description)
      return { success: true, spec: analysis }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }, [])

  return {
    handleSendMessage,
    handleValidateDescription,
    handlePreviewModel,
  }
}

