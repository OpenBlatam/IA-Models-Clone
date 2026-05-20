/**
 * Backup Manager
 * Sistema de backup automático de cola, modelos y configuraciones
 */

import { QueueItem } from '../components/ProactiveModelBuilder'
import { ProactiveBuildResult } from '../components/ProactiveModelBuilder'

export interface BackupData {
  version: string
  timestamp: number
  queue: QueueItem[]
  completedModels: ProactiveBuildResult[]
  settings?: {
    autoMode?: boolean
    batchMode?: boolean
    batchConcurrency?: number
    notificationsEnabled?: boolean
  }
}

export class BackupManager {
  private backupInterval: number = 5 * 60 * 1000 // 5 minutos
  private maxBackups: number = 10
  private storageKey: string = 'proactive-builder-backups'
  private intervalId: NodeJS.Timeout | null = null

  /**
   * Iniciar backup automático
   */
  startAutoBackup(
    data: {
      queue?: QueueItem[]
      models?: ProactiveBuildResult[]
      completedModels?: ProactiveBuildResult[]
      settings?: Record<string, any>
    } | (() => QueueItem[]),
    getCompletedModels?: () => ProactiveBuildResult[],
    getSettings?: () => Record<string, any>,
    interval?: number
  ): (() => void) {
    if (this.intervalId) {
      this.stopAutoBackup()
    }

    // Handle object parameter
    if (typeof data === 'object' && !Array.isArray(data) && typeof data !== 'function') {
      const backupInterval = interval || this.backupInterval
      this.intervalId = setInterval(() => {
        try {
          const backup = this.createBackup(data)
          this.saveBackup(backup)
          this.cleanupOldBackups()
        } catch (error) {
          console.error('Error en backup automático:', error)
        }
      }, backupInterval)

      return () => {
        this.stopAutoBackup()
      }
    }

    // Handle function parameters (legacy)
    const getQueue = data as () => QueueItem[]
    const getCompleted = getCompletedModels || (() => [] as ProactiveBuildResult[])
    const getSettingsFn = getSettings

    this.intervalId = setInterval(() => {
      try {
        const backup = this.createBackup(
          getQueue(),
          getCompleted(),
          getSettingsFn?.()
        )
        this.saveBackup(backup)
        this.cleanupOldBackups()
      } catch (error) {
        console.error('Error en backup automático:', error)
      }
    }, interval || this.backupInterval)

    // Retornar función de cleanup
    return () => {
      this.stopAutoBackup()
    }
  }

  /**
   * Detener backup automático
   */
  stopAutoBackup(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  /**
   * Crear backup manual
   */
  createBackup(
    data: {
      queue?: QueueItem[]
      models?: ProactiveBuildResult[]
      completedModels?: ProactiveBuildResult[]
      settings?: Record<string, any>
      [key: string]: any
    } | QueueItem[],
    completedModels?: ProactiveBuildResult[],
    settings?: Record<string, any>
  ): BackupData {
    // Handle object parameter (new signature)
    if (typeof data === 'object' && !Array.isArray(data)) {
      const queue = Array.isArray(data.queue) ? data.queue : []
      const models = Array.isArray(data.models) ? data.models : []
      const completed = Array.isArray(data.completedModels) ? data.completedModels : models
      
      return {
        version: '1.0.0',
        id: `backup-${Date.now()}`,
        timestamp: Date.now(),
        queue: [...queue],
        completedModels: [...completed],
        settings: data.settings ? { ...data.settings } : undefined,
      } as BackupData & { id: string }
    }
    
    // Handle legacy signature (array parameters)
    const queue = Array.isArray(data) ? data : []
    const completed = Array.isArray(completedModels) ? completedModels : []
    
    return {
      version: '1.0.0',
      id: `backup-${Date.now()}`,
      timestamp: Date.now(),
      queue: [...queue],
      completedModels: [...completed],
      settings: settings ? { ...settings } : undefined,
    } as BackupData & { id: string }
  }

  /**
   * Guardar backup
   */
  saveBackup(backup: BackupData): void {
    try {
      const backups = this.getAllBackups()
      backups.push(backup)
      
      // Ordenar por timestamp (más reciente primero)
      backups.sort((a, b) => b.timestamp - a.timestamp)
      
      // Mantener solo los últimos N backups
      const limitedBackups = backups.slice(0, this.maxBackups)
      
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(
          this.storageKey,
          JSON.stringify(limitedBackups)
        )
      }
    } catch (error) {
      console.error('Error guardando backup:', error)
    }
  }

  /**
   * Obtener todos los backups
   */
  getAllBackups(): BackupData[] {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const data = window.localStorage.getItem(this.storageKey)
        if (data) {
          return JSON.parse(data)
        }
      }
    } catch (error) {
      console.error('Error obteniendo backups:', error)
    }
    return []
  }

  /**
   * Obtener backup por ID o timestamp
   */
  getBackup(idOrTimestamp: string | number): BackupData | null {
    const backups = this.getAllBackups()
    const id = typeof idOrTimestamp === 'string' ? idOrTimestamp : idOrTimestamp.toString()
    const timestamp = typeof idOrTimestamp === 'number' ? idOrTimestamp : undefined
    
    return backups.find(b => 
      (b as any).id === id || 
      b.timestamp.toString() === id ||
      (timestamp && b.timestamp === timestamp)
    ) || null
  }

  /**
   * Restaurar backup por ID
   */
  restoreBackup(idOrTimestamp: string | number): {
    queue: QueueItem[]
    completedModels: ProactiveBuildResult[]
    settings?: Record<string, any>
  } | null {
    const backup = this.getBackup(idOrTimestamp)
    if (!backup) return null
    return this.restoreFromBackup(backup)
  }

  /**
   * Obtener backup más reciente
   */
  getLatestBackup(): BackupData | null {
    const backups = this.getAllBackups()
    return backups.length > 0 ? backups[0] : null
  }

  /**
   * Restaurar desde backup
   */
  restoreFromBackup(backup: BackupData): {
    queue: QueueItem[]
    completedModels: ProactiveBuildResult[]
    settings?: Record<string, any>
  } {
    return {
      queue: [...backup.queue],
      completedModels: [...backup.completedModels],
      settings: backup.settings ? { ...backup.settings } : undefined,
    }
  }

  /**
   * Exportar backup
   */
  exportBackup(backupIdOrData: string | BackupData): string {
    let backup: BackupData
    if (typeof backupIdOrData === 'string') {
      const found = this.getBackup(backupIdOrData)
      if (!found) {
        throw new Error('Backup not found')
      }
      backup = found
    } else {
      backup = backupIdOrData
    }
    
    return JSON.stringify(backup, null, 2)
  }

  /**
   * Importar backup
   */
  importBackup(json: string): BackupData {
    const data = JSON.parse(json)
    if (this.validateBackup(data)) {
      return data
    } else {
      throw new Error('Formato de backup inválido')
    }
  }

  /**
   * Validar backup
   */
  private validateBackup(data: any): data is BackupData {
    return (
      data &&
      typeof data === 'object' &&
      typeof data.version === 'string' &&
      typeof data.timestamp === 'number' &&
      Array.isArray(data.queue) &&
      Array.isArray(data.completedModels)
    )
  }

  /**
   * Limpiar backups antiguos
   */
  private cleanupOldBackups(): void {
    const backups = this.getAllBackups()
    if (backups.length > this.maxBackups) {
      const limitedBackups = backups.slice(0, this.maxBackups)
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(
          this.storageKey,
          JSON.stringify(limitedBackups)
        )
      }
    }
  }

  /**
   * Limpiar todos los backups
   */
  clearAllBackups(): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(this.storageKey)
    }
  }

  /**
   * Limpiar (alias para compatibilidad)
   */
  clear(): void {
    this.clearAllBackups()
  }

  /**
   * Configurar intervalo de backup
   */
  setBackupInterval(interval: number): void {
    this.backupInterval = interval
  }

  /**
   * Configurar máximo de backups
   */
  setMaxBackups(max: number): void {
    this.maxBackups = max
  }
}

// Singleton instance
let backupManagerInstance: BackupManager | null = null

export function getBackupManager(): BackupManager {
  if (!backupManagerInstance) {
    backupManagerInstance = new BackupManager()
  }
  return backupManagerInstance
}

export default BackupManager

