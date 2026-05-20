/**
 * Model Versioning
 * Sistema de versionado de modelos
 */

import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export interface ModelVersion {
  version: string
  modelId: string
  modelName: string
  description: string
  config: any
  status: 'completed' | 'failed'
  createdAt: number
  parentVersion?: string
  changes?: string[]
  performance?: {
    duration: number
    metrics?: Record<string, number>
  }
}

export class ModelVersioning {
  private versions: Map<string, ModelVersion[]> = new Map()
  private versionFormat: 'semantic' | 'timestamp' = 'semantic'

  /**
   * Crear nueva versión
   */
  createVersion(
    modelName: string,
    modelData: ProactiveBuildResult,
    config: any,
    parentVersion?: string
  ): ModelVersion {
    const existingVersions = this.versions.get(modelName) || []
    const version = this.generateVersion(existingVersions.length, modelData.createdAt || Date.now())

    const newVersion: ModelVersion = {
      version,
      modelId: modelData.modelId,
      modelName,
      description: modelData.description,
      config,
      status: modelData.status,
      createdAt: modelData.endTime || Date.now(),
      parentVersion,
      performance: modelData.duration
        ? {
            duration: modelData.duration,
          }
        : undefined,
    }

    const versions = [...existingVersions, newVersion]
    this.versions.set(modelName, versions)

    return newVersion
  }

  /**
   * Generar número de versión
   */
  private generateVersion(index: number, timestamp: number): string {
    if (this.versionFormat === 'semantic') {
      return `v${index + 1}.0.0`
    } else {
      return `v${timestamp}`
    }
  }

  /**
   * Obtener versiones de un modelo
   */
  getVersions(modelName: string): ModelVersion[] {
    return this.versions.get(modelName) || []
  }

  /**
   * Obtener última versión
   */
  getLatestVersion(modelName: string): ModelVersion | null {
    const versions = this.getVersions(modelName)
    return versions.length > 0 ? versions[versions.length - 1] : null
  }

  /**
   * Obtener versión específica
   */
  getVersion(modelName: string, version: string): ModelVersion | null {
    const versions = this.getVersions(modelName)
    return versions.find(v => v.version === version) || null
  }

  /**
   * Comparar versiones
   */
  compareVersions(modelName: string, version1: string, version2: string): {
    differences: string[]
    performanceDiff?: {
      duration: number
      metrics?: Record<string, number>
    }
  } {
    const v1 = this.getVersion(modelName, version1)
    const v2 = this.getVersion(modelName, version2)

    if (!v1 || !v2) {
      return { differences: [] }
    }

    const differences: string[] = []

    // Comparar configuraciones
    if (JSON.stringify(v1.config) !== JSON.stringify(v2.config)) {
      differences.push('Configuración cambiada')
    }

    // Comparar performance
    if (v1.performance && v2.performance) {
      const durationDiff = v2.performance.duration - v1.performance.duration
      if (Math.abs(durationDiff) > 1000) {
        differences.push(`Duración cambiada: ${Math.round(durationDiff / 1000)}s`)
      }
    }

    return {
      differences,
      performanceDiff: v1.performance && v2.performance
        ? {
            duration: v2.performance.duration - v1.performance.duration,
          }
        : undefined,
    }
  }

  /**
   * Obtener historial de versiones
   */
  getVersionHistory(modelName: string): ModelVersion[] {
    return this.getVersions(modelName)
  }

  /**
   * Exportar versiones
   */
  exportVersions(modelName: string): string {
    const versions = this.getVersions(modelName)
    return JSON.stringify(versions, null, 2)
  }

  /**
   * Establecer formato de versión
   */
  setVersionFormat(format: 'semantic' | 'timestamp'): void {
    this.versionFormat = format
  }
}

// Singleton instance
let versioningInstance: ModelVersioning | null = null

export function getModelVersioning(): ModelVersioning {
  if (!versioningInstance) {
    versioningInstance = new ModelVersioning()
  }
  return versioningInstance
}

export default ModelVersioning










