'use client';

import { useState } from 'react';
import { FiSearch, FiX } from 'react-icons/fi';
import { SearchHistoryDropdown, useSearchHistory } from './SearchHistory';

interface SearchBarProps {
  placeholder?: string;
  onSearch: (query: string) => void;
  className?: string;
}

export default function SearchBar({ placeholder = 'Buscar...', onSearch, className = '' }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const { addToHistory } = useSearchHistory();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      addToHistory(query);
    }
    onSearch(query);
    setShowHistory(false);
  };

  const handleClear = () => {
    setQuery('');
    onSearch('');
    setShowHistory(false);
  };

  const handleSelect = (selectedQuery: string) => {
    setQuery(selectedQuery);
    onSearch(selectedQuery);
    setShowHistory(false);
  };

  return (
    <form onSubmit={handleSubmit} className={`relative ${className}`}>
      <div className="relative">
        <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            if (e.target.value === '') {
              onSearch('');
            }
          }}
          onFocus={() => setShowHistory(true)}
          placeholder={placeholder}
          className="w-full pl-10 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        />
        {query && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <FiX size={20} />
          </button>
        )}
        <SearchHistoryDropdown onSelect={handleSelect} isVisible={showHistory && !query} />
      </div>
    </form>
  );
}

