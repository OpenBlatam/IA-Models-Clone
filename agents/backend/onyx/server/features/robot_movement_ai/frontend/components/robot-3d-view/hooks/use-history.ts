/**
 * Hook for history management (undo/redo)
 * @module robot-3d-view/hooks/use-history
 */

import { useState, useEffect, useCallback } from 'react';
import { historyManager, type HistoryEntry } from '../lib/history-manager';
import type { SceneConfig } from '../schemas/validation-schemas';

/**
 * Hook for managing configuration history
 * 
 * Provides undo/redo functionality for configuration changes.
 * 
 * @returns History state and actions
 * 
 * @example
 * ```tsx
 * const { canUndo, canRedo, undo, redo } = useHistory();
 * ```
 */
export function useHistory() {
  const [canUndo, setCanUndo] = useState(() => historyManager.canUndo());
  const [canRedo, setCanRedo] = useState(() => historyManager.canRedo());
  const [history, setHistory] = useState<readonly HistoryEntry[]>(() =>
    historyManager.getHistory()
  );

  useEffect(() => {
    const updateState = () => {
      setCanUndo(historyManager.canUndo());
      setCanRedo(historyManager.canRedo());
      setHistory(historyManager.getHistory());
    };

    // Update state periodically (could be improved with event system)
    const interval = setInterval(updateState, 100);
    return () => clearInterval(interval);
  }, []);

  const undo = useCallback((): SceneConfig | null => {
    const config = historyManager.undo();
    setCanUndo(historyManager.canUndo());
    setCanRedo(historyManager.canRedo());
    setHistory(historyManager.getHistory());
    return config;
  }, []);

  const redo = useCallback((): SceneConfig | null => {
    const config = historyManager.redo();
    setCanUndo(historyManager.canUndo());
    setCanRedo(historyManager.canRedo());
    setHistory(historyManager.getHistory());
    return config;
  }, []);

  const addEntry = useCallback((config: SceneConfig, description?: string) => {
    historyManager.addEntry(config, description);
    setCanUndo(historyManager.canUndo());
    setCanRedo(historyManager.canRedo());
    setHistory(historyManager.getHistory());
  }, []);

  const clear = useCallback(() => {
    historyManager.clear();
    setCanUndo(false);
    setCanRedo(false);
    setHistory([]);
  }, []);

  return {
    canUndo,
    canRedo,
    undo,
    redo,
    addEntry,
    clear,
    history,
  };
}



