/**
 * Backup and restore system
 * @module robot-3d-view/utils/backup-restore
 */

import type { SceneConfig } from '../schemas/validation-schemas';
import { compressionManager } from './compression';

/**
 * Backup entry
 */
export interface BackupEntry {
  id: string;
  name: string;
  timestamp: number;
  data: SceneConfig;
  size: number;
  compressed?: boolean;
}

/**
 * Backup Manager class
 */
export class BackupManager {
  private backups: Map<string, BackupEntry> = new Map();
  private maxBackups = 50;
  private autoBackupInterval: NodeJS.Timeout | null = null;
  private autoBackupEnabled = false;
  private autoBackupIntervalMs = 300000; // 5 minutes

  /**
   * Creates a backup
   */
  async createBackup(
    name: string,
    data: SceneConfig,
    compress = false
  ): Promise<string> {
    const id = `backup-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const json = JSON.stringify(data);
    let size = json.length;
    let compressed = false;

    if (compress) {
      try {
        const compressedData = await compressionManager.compressJSON(data);
        size = compressedData.length;
        compressed = true;
      } catch (error) {
        console.warn('Compression failed, storing uncompressed:', error);
      }
    }

    const backup: BackupEntry = {
      id,
      name,
      timestamp: Date.now(),
      data,
      size,
      compressed,
    };

    this.backups.set(id, backup);

    // Limit backups
    if (this.backups.size > this.maxBackups) {
      this.evictOldest();
    }

    // Store in localStorage
    this.saveToLocalStorage();

    return id;
  }

  /**
   * Restores a backup
   */
  restoreBackup(id: string): SceneConfig | null {
    const backup = this.backups.get(id);
    if (!backup) return null;

    return backup.data;
  }

  /**
   * Deletes a backup
   */
  deleteBackup(id: string): boolean {
    const deleted = this.backups.delete(id);
    if (deleted) {
      this.saveToLocalStorage();
    }
    return deleted;
  }

  /**
   * Gets all backups
   */
  getAllBackups(): BackupEntry[] {
    return Array.from(this.backups.values()).sort(
      (a, b) => b.timestamp - a.timestamp
    );
  }

  /**
   * Gets a backup
   */
  getBackup(id: string): BackupEntry | undefined {
    return this.backups.get(id);
  }

  /**
   * Exports a backup
   */
  async exportBackup(id: string, compress = false): Promise<Blob> {
    const backup = this.backups.get(id);
    if (!backup) {
      throw new Error(`Backup not found: ${id}`);
    }

    const json = JSON.stringify(backup, null, 2);

    if (compress) {
      const compressed = await compressionManager.compress(json);
      return new Blob([compressed], { type: 'application/gzip' });
    }

    return new Blob([json], { type: 'application/json' });
  }

  /**
   * Imports a backup
   */
  async importBackup(file: File): Promise<string> {
    const text = await file.text();
    let backup: BackupEntry;

    try {
      // Try to decompress
      const arrayBuffer = await file.arrayBuffer();
      const decompressed = await compressionManager.decompress(
        new Uint8Array(arrayBuffer)
      );
      backup = JSON.parse(decompressed);
    } catch {
      // Not compressed, parse directly
      backup = JSON.parse(text);
    }

    this.backups.set(backup.id, backup);
    this.saveToLocalStorage();

    return backup.id;
  }

  /**
   * Enables auto-backup
   */
  enableAutoBackup(
    getCurrentConfig: () => SceneConfig,
    intervalMs = 300000
  ): void {
    this.autoBackupEnabled = true;
    this.autoBackupIntervalMs = intervalMs;

    if (this.autoBackupInterval) {
      clearInterval(this.autoBackupInterval);
    }

    this.autoBackupInterval = setInterval(() => {
      const config = getCurrentConfig();
      this.createBackup(`Auto-backup ${new Date().toISOString()}`, config, true);
    }, intervalMs);
  }

  /**
   * Disables auto-backup
   */
  disableAutoBackup(): void {
    this.autoBackupEnabled = false;
    if (this.autoBackupInterval) {
      clearInterval(this.autoBackupInterval);
      this.autoBackupInterval = null;
    }
  }

  /**
   * Evicts oldest backup
   */
  private evictOldest(): void {
    const backups = Array.from(this.backups.values());
    const oldest = backups.sort((a, b) => a.timestamp - b.timestamp)[0];
    if (oldest) {
      this.backups.delete(oldest.id);
    }
  }

  /**
   * Saves backups to localStorage
   */
  private saveToLocalStorage(): void {
    try {
      const backups = Array.from(this.backups.values());
      const data = JSON.stringify(backups);
      localStorage.setItem('robot-3d-view-backups', data);
    } catch (error) {
      console.warn('Failed to save backups to localStorage:', error);
    }
  }

  /**
   * Loads backups from localStorage
   */
  loadFromLocalStorage(): void {
    try {
      const data = localStorage.getItem('robot-3d-view-backups');
      if (data) {
        const backups = JSON.parse(data) as BackupEntry[];
        backups.forEach((backup) => {
          this.backups.set(backup.id, backup);
        });
      }
    } catch (error) {
      console.warn('Failed to load backups from localStorage:', error);
    }
  }

  /**
   * Clears all backups
   */
  clear(): void {
    this.backups.clear();
    localStorage.removeItem('robot-3d-view-backups');
  }

  /**
   * Gets backup statistics
   */
  getStats(): {
    total: number;
    totalSize: number;
    averageSize: number;
    oldest: number | null;
    newest: number | null;
    autoBackupEnabled: boolean;
  } {
    const backups = Array.from(this.backups.values());
    const totalSize = backups.reduce((sum, b) => sum + b.size, 0);
    const timestamps = backups.map((b) => b.timestamp);

    return {
      total: backups.length,
      totalSize,
      averageSize: backups.length > 0 ? totalSize / backups.length : 0,
      oldest: timestamps.length > 0 ? Math.min(...timestamps) : null,
      newest: timestamps.length > 0 ? Math.max(...timestamps) : null,
      autoBackupEnabled: this.autoBackupEnabled,
    };
  }
}

/**
 * Global backup manager instance
 */
export const backupManager = new BackupManager();

// Load backups on initialization
backupManager.loadFromLocalStorage();



