/**
 * History Manager for configuration changes
 * 
 * Provides undo/redo functionality for configuration changes.
 * 
 * @module robot-3d-view/lib/history-manager
 */

import type { SceneConfig } from '../schemas/validation-schemas';

/**
 * History entry
 */
export interface HistoryEntry {
  config: SceneConfig;
  timestamp: number;
  description?: string;
}

/**
 * History Manager class
 */
export class HistoryManager {
  private history: HistoryEntry[] = [];
  private currentIndex = -1;
  private maxHistorySize = 50;

  /**
   * Adds a new entry to history
   */
  addEntry(config: SceneConfig, description?: string): void {
    // Remove any entries after current index (when undoing and making new changes)
    this.history = this.history.slice(0, this.currentIndex + 1);

    // Add new entry
    this.history.push({
      config: { ...config },
      timestamp: Date.now(),
      description,
    });

    // Limit history size
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    } else {
      this.currentIndex++;
    }
  }

  /**
   * Gets current configuration
   */
  getCurrent(): SceneConfig | null {
    if (this.currentIndex < 0 || this.currentIndex >= this.history.length) {
      return null;
    }
    return { ...this.history[this.currentIndex].config };
  }

  /**
   * Undo last change
   */
  undo(): SceneConfig | null {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      return this.getCurrent();
    }
    return null;
  }

  /**
   * Redo last undone change
   */
  redo(): SceneConfig | null {
    if (this.currentIndex < this.history.length - 1) {
      this.currentIndex++;
      return this.getCurrent();
    }
    return null;
  }

  /**
   * Checks if undo is available
   */
  canUndo(): boolean {
    return this.currentIndex > 0;
  }

  /**
   * Checks if redo is available
   */
  canRedo(): boolean {
    return this.currentIndex < this.history.length - 1;
  }

  /**
   * Clears history
   */
  clear(): void {
    this.history = [];
    this.currentIndex = -1;
  }

  /**
   * Gets history entries
   */
  getHistory(): readonly HistoryEntry[] {
    return [...this.history];
  }
}

/**
 * Global history manager instance
 */
export const historyManager = new HistoryManager();



