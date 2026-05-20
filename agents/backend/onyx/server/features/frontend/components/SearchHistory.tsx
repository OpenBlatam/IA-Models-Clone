'use client';

import { useState, useEffect } from 'react';
import { FiClock, FiX, FiSearch } from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';

interface SearchHistoryItem {
  id: string;
  query: string;
  timestamp: Date;
}

export function useSearchHistory() {
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('bul_search_history');
    if (stored) {
      setHistory(JSON.parse(stored).map((item: any) => ({
        ...item,
        timestamp: new Date(item.timestamp),
      })));
    }
  }, []);

  const addToHistory = (query: string) => {
    if (!query.trim()) return;

    const newItem: SearchHistoryItem = {
      id: Date.now().toString(),
      query: query.trim(),
      timestamp: new Date(),
    };

    const updated = [
      newItem,
      ...history.filter((item) => item.query.toLowerCase() !== query.trim().toLowerCase()),
    ].slice(0, 10); // Keep only last 10

    setHistory(updated);
    localStorage.setItem('bul_search_history', JSON.stringify(updated));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.setItem('bul_search_history', JSON.stringify([]));
  };

  const removeItem = (id: string) => {
    const updated = history.filter((item) => item.id !== id);
    setHistory(updated);
    localStorage.setItem('bul_search_history', JSON.stringify(updated));
  };

  return { history, addToHistory, clearHistory, removeItem };
}

interface SearchHistoryDropdownProps {
  onSelect: (query: string) => void;
  isVisible: boolean;
}

export function SearchHistoryDropdown({ onSelect, isVisible }: SearchHistoryDropdownProps) {
  const { history, clearHistory, removeItem } = useSearchHistory();

  if (!isVisible || history.length === 0) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-50 max-h-64 overflow-y-auto"
      >
        <div className="p-2">
          <div className="flex items-center justify-between px-3 py-2 mb-2">
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
              <FiClock size={16} />
              <span>Búsquedas recientes</span>
            </div>
            <button
              onClick={clearHistory}
              className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            >
              Limpiar
            </button>
          </div>
          {history.map((item) => (
            <button
              key={item.id}
              onClick={() => onSelect(item.query)}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg transition-colors group"
            >
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <FiSearch size={16} className="text-gray-400 flex-shrink-0" />
                <span className="text-sm text-gray-700 dark:text-gray-300 truncate">
                  {item.query}
                </span>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeItem(item.id);
                }}
                className="opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <FiX size={16} className="text-gray-400 hover:text-gray-600" />
              </button>
            </button>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


