'use client';

import { useState, useEffect } from 'react';
import { Clock, X, Search } from 'lucide-react';

interface SearchHistoryProps {
  onSelect: (query: string) => void;
}

export function SearchHistory({ onSelect }: SearchHistoryProps) {
  const [history, setHistory] = useState<string[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem('search-history');
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  const removeFromHistory = (query: string) => {
    const updated = history.filter((h) => h !== query);
    setHistory(updated);
    localStorage.setItem('search-history', JSON.stringify(updated));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('search-history');
  };

  if (history.length === 0) return null;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-gray-400" />
          <h3 className="text-sm font-semibold text-white">Historial de Búsqueda</h3>
        </div>
        <button
          onClick={clearHistory}
          className="text-xs text-gray-400 hover:text-white"
        >
          Limpiar
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {history.slice(0, 10).map((query, idx) => (
          <div
            key={idx}
            className="flex items-center gap-1 px-3 py-1 bg-white/5 hover:bg-white/10 rounded-lg transition-colors group"
          >
            <button
              onClick={() => onSelect(query)}
              className="text-sm text-gray-300 hover:text-white"
            >
              {query}
            </button>
            <button
              onClick={() => removeFromHistory(query)}
              className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-300 transition-opacity"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}


