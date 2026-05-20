import * as fs from 'fs/promises'
import * as path from 'path'

/**
 * Verifica que el directorio de TruthGPT existe
 */
export async function verifyTruthGPTPath(truthgptPath: string): Promise<boolean> {
  try {
    await fs.access(truthgptPath)
    return true
  } catch {
    return false
  }
}

/**
 * Crea el directorio de modelos generados si no existe
 */
export async function ensureGeneratedModelsDir(truthgptPath: string): Promise<string> {
  const modelsDir = path.join(truthgptPath, 'generated_models')
  try {
    await fs.mkdir(modelsDir, { recursive: true })
  } catch (error) {
    console.error('Error creating generated_models directory:', error)
    throw error
  }
  return modelsDir
}

/**
 * Sanitiza el nombre del modelo para usarlo como nombre de repositorio
 */
export function sanitizeModelName(input: string): string {
  return input
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-')
    .substring(0, 50)
    .replace(/^-+|-+$/g, '')
}

/**
 * Genera un ID único para el modelo
 */
export function generateModelId(): string {
  return `model-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}


