import { useState, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../types';
import ApiService from '../services/apiService';
import { HistoryItem } from '../types';

interface UseHistoryReturn {
  history: HistoryItem[];
  isLoading: boolean;
  error: string | null;
  loadHistory: () => Promise<void>;
  refreshHistory: () => Promise<void>;
  addToHistory: (item: HistoryItem) => void;
  clearHistory: () => void;
}

export const useHistory = (): UseHistoryReturn => {
  const dispatch = useDispatch();
  const { history, isLoading, error } = useSelector((state: RootState) => state.history);
  const { userId } = useSelector((state: RootState) => state.user);

  const loadHistory = useCallback(async () => {
    try {
      dispatch({ type: 'HISTORY_LOAD_START' });
      const result = await ApiService.getHistory(userId || 'default_user');
      const historyData = result.data?.history || result.history || result || [];
      dispatch({
        type: 'HISTORY_LOAD_SUCCESS',
        payload: Array.isArray(historyData) ? historyData : [],
      });
    } catch (err: any) {
      console.error('Error loading history:', err);
      dispatch({
        type: 'HISTORY_LOAD_FAILURE',
        payload: err.message || 'Error al cargar el historial',
      });
    }
  }, [userId, dispatch]);

  const refreshHistory = useCallback(async () => {
    await loadHistory();
  }, [loadHistory]);

  const addToHistory = useCallback((item: HistoryItem) => {
    dispatch({
      type: 'ADD_TO_HISTORY',
      payload: item,
    });
  }, [dispatch]);

  const clearHistory = useCallback(() => {
    dispatch({
      type: 'HISTORY_LOAD_SUCCESS',
      payload: [],
    });
  }, [dispatch]);

  return {
    history,
    isLoading,
    error,
    loadHistory,
    refreshHistory,
    addToHistory,
    clearHistory,
  };
};

