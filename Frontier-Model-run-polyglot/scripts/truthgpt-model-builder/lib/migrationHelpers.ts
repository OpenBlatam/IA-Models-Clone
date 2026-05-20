/**
 * Helpers para migrar código existente al nuevo sistema
 * ======================================================
 */

/**
 * Migra la función createModel antigua al nuevo sistema
 */
export function migrateCreateModel(
  oldCreateModel: (description: string, spec?: any) => Promise<any>,
  newSystem: ReturnType<typeof useCompleteModelSystem>
) {
  return async (description: string, spec?: any) => {
    // Usar el nuevo sistema
    const modelId = await newSystem.createModel({
      description,
      spec,
      enableOptimization: true,
      enableValidation: true
    })

    if (modelId) {
      // Retornar en formato compatible con el código antiguo
      return {
        modelId,
        modelName: spec?.modelName || 'Model',
        description,
        status: 'creating',
        githubUrl: null
      }
    }

    return null
  }
}

/**
 * Migra la función pollModelStatus antigua al nuevo sistema
 */
export function migratePollModelStatus(
  oldPollModelStatus: (modelId: string) => void,
  newSystem: ReturnType<typeof useCompleteModelSystem>
) {
  return (modelId: string) => {
    // El nuevo sistema maneja el polling automáticamente
    // Esta función existe solo para compatibilidad
    if (newSystem.activeModels.has(modelId)) {
      // El modelo ya está siendo monitoreado
      return
    }

    // Si el modelo no está en el sistema, agregarlo al historial si existe
    const modelInfo = newSystem.history.find(m => m.id === modelId)
    if (modelInfo && modelInfo.status === 'creating') {
      // El sistema ya lo está monitoreando a través de createModel
      return
    }
  }
}

/**
 * Crea un wrapper de compatibilidad para el código antiguo
 */
export function createCompatibilityWrapper(
  newSystem: ReturnType<typeof useCompleteModelSystem>
) {
  return {
    // Compatibilidad con createModel antiguo
    createModel: async (description: string, spec?: any) => {
      return await migrateCreateModel(
        async (desc, s) => ({ modelId: '', modelName: '', description: desc }),
        newSystem
      )(description, spec)
    },

    // Compatibilidad con pollModelStatus antiguo
    pollModelStatus: (modelId: string) => {
      migratePollModelStatus(() => {}, newSystem)(modelId)
    },

    // Nuevas funcionalidades disponibles
    ...newSystem
  }
}










