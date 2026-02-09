import { useState, useEffect } from 'react';
import { storage, STORAGE_KEYS } from '../utils/storage';

export interface ActionHistoryItem {
  id: string;
  type: 'create' | 'delete' | 'export' | 'validate' | 'favorite' | 'share';
  projectId?: string;
  projectName?: string;
  timestamp: number;
  success: boolean;
}

const MAX_HISTORY_ITEMS = 50;

export const useActionHistory = () => {
  const [history, setHistory] = useState<ActionHistoryItem[]>([]);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const saved = await storage.get<ActionHistoryItem[]>(STORAGE_KEYS.ACTION_HISTORY) || [];
      setHistory(saved);
    } catch (error) {
      console.error('Error loading action history:', error);
    }
  };

  const addAction = async (action: Omit<ActionHistoryItem, 'id' | 'timestamp'>) => {
    try {
      const newAction: ActionHistoryItem = {
        ...action,
        id: `${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
      };

      const updated = [newAction, ...history].slice(0, MAX_HISTORY_ITEMS);
      setHistory(updated);
      await storage.set(STORAGE_KEYS.ACTION_HISTORY, updated);
    } catch (error) {
      console.error('Error saving action history:', error);
    }
  };

  const clearHistory = async () => {
    try {
      setHistory([]);
      await storage.remove(STORAGE_KEYS.ACTION_HISTORY);
    } catch (error) {
      console.error('Error clearing action history:', error);
    }
  };

  return {
    history,
    addAction,
    clearHistory,
  };
};

