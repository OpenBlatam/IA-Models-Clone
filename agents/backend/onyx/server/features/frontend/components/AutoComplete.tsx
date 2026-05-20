'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiZap } from 'react-icons/fi';

interface Suggestion {
  text: string;
  type: 'template' | 'history' | 'suggestion';
}

interface AutoCompleteProps {
  value: string;
  onChange: (value: string) => void;
  onSelect?: (value: string) => void;
  suggestions?: string[];
  placeholder?: string;
  className?: string;
}

const commonSuggestions = [
  'plan de marketing',
  'propuesta comercial',
  'reporte financiero',
  'estrategia tecnológica',
  'política de recursos humanos',
  'manual de operaciones',
  'análisis de mercado',
  'plan de negocios',
  'documento técnico',
  'guía de usuario',
];

export default function AutoComplete({
  value,
  onChange,
  onSelect,
  suggestions = [],
  placeholder = 'Escribe tu consulta...',
  className = '',
}: AutoCompleteProps) {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [filteredSuggestions, setFilteredSuggestions] = useState<Suggestion[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (value.length > 2) {
      const allSuggestions: Suggestion[] = [
        ...suggestions.map((s) => ({ text: s, type: 'suggestion' as const })),
        ...commonSuggestions
          .filter((s) => s.toLowerCase().includes(value.toLowerCase()))
          .map((s) => ({ text: s, type: 'suggestion' as const })),
      ].slice(0, 5);

      setFilteredSuggestions(allSuggestions);
      setShowSuggestions(allSuggestions.length > 0);
    } else {
      setShowSuggestions(false);
    }
  }, [value, suggestions]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (suggestion: string) => {
    onChange(suggestion);
    onSelect?.(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || filteredSuggestions.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < filteredSuggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      handleSelect(filteredSuggestions[selectedIndex].text);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <textarea
        ref={inputRef}
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setSelectedIndex(-1);
        }}
        onFocus={() => value.length > 2 && setShowSuggestions(true)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="textarea"
        rows={6}
      />

      <AnimatePresence>
        {showSuggestions && filteredSuggestions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-50 max-h-64 overflow-y-auto"
          >
            <div className="p-2">
              <div className="px-3 py-2 text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                Sugerencias
              </div>
              {filteredSuggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSelect(suggestion.text)}
                  className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors text-left ${
                    index === selectedIndex
                      ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  <FiZap size={16} className="text-primary-500 flex-shrink-0" />
                  <span className="text-sm">{suggestion.text}</span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

