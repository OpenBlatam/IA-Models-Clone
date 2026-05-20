'use client';

import { useState, useEffect } from 'react';
import { Clock, TrendingUp } from 'lucide-react';

interface MusicRecentSearchesProps {
  onSelect: (query: string) => void;
}

export function MusicRecentSearches({ onSelect }: MusicRecentSearchesProps) {
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem('recent-searches');
    if (saved) {
      try {
        setRecentSearches(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading recent searches:', e);
      }
    }
  }, []);

  if (recentSearches.length === 0) {
    return null;
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="w-4 h-4 text-purple-300" />
        <h3 className="text-sm font-semibold text-white">Búsquedas Recientes</h3>
      </div>

      <div className="flex flex-wrap gap-2">
        {recentSearches.slice(0, 8).map((query, idx) => (
          <button
            key={idx}
            onClick={() => onSelect(query)}
            className="px-3 py-1 bg-white/5 hover:bg-white/10 rounded-lg text-sm text-gray-300 hover:text-white transition-colors flex items-center gap-1"
          >
            <Clock className="w-3 h-3" />
            {query}
          </button>
        ))}
      </div>
    </div>
  );
}


