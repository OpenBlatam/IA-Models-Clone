'use client';

import { useState, useEffect } from 'react';
import { Clock, TrendingUp } from 'lucide-react';

interface SearchSuggestionsProps {
  onSelect: (query: string) => void;
}

export function SearchSuggestions({ onSelect }: SearchSuggestionsProps) {
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [trendingSearches] = useState<string[]>([
    'Bohemian Rhapsody',
    'Blinding Lights',
    'Shape of You',
    'Watermelon Sugar',
    'Levitating',
  ]);

  useEffect(() => {
    const saved = localStorage.getItem('recent-searches');
    if (saved) {
      setRecentSearches(JSON.parse(saved));
    }
  }, []);

  const handleSelect = (query: string) => {
    const updated = [query, ...recentSearches.filter((s) => s !== query)].slice(0, 5);
    setRecentSearches(updated);
    localStorage.setItem('recent-searches', JSON.stringify(updated));
    onSelect(query);
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="grid md:grid-cols-2 gap-4">
        {recentSearches.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-gray-400" />
              <h3 className="text-sm font-semibold text-white">Búsquedas Recientes</h3>
            </div>
            <div className="space-y-1">
              {recentSearches.map((search, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelect(search)}
                  className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-white/10 rounded-lg transition-colors"
                >
                  {search}
                </button>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-gray-400" />
            <h3 className="text-sm font-semibold text-white">Tendencias</h3>
          </div>
          <div className="space-y-1">
            {trendingSearches.map((search, idx) => (
              <button
                key={idx}
                onClick={() => handleSelect(search)}
                className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-white/10 rounded-lg transition-colors"
              >
                {search}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}


