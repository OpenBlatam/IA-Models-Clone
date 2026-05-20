'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiTag, FiX, FiPlus } from 'react-icons/fi';

interface TagsManagerProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  suggestions?: string[];
}

const defaultSuggestions = [
  'importante',
  'urgente',
  'revisar',
  'finalizado',
  'borrador',
  'marketing',
  'ventas',
  'tecnología',
  'finanzas',
  'recursos-humanos',
];

export default function TagsManager({
  tags,
  onChange,
  suggestions = defaultSuggestions,
}: TagsManagerProps) {
  const [inputValue, setInputValue] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const availableSuggestions = suggestions.filter(
    (s) => !tags.includes(s) && s.toLowerCase().includes(inputValue.toLowerCase())
  );

  const addTag = (tag: string) => {
    if (tag.trim() && !tags.includes(tag.trim())) {
      onChange([...tags, tag.trim()]);
      setInputValue('');
      setShowSuggestions(false);
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(tags.filter((tag) => tag !== tagToRemove));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      addTag(inputValue);
    } else if (e.key === 'Backspace' && !inputValue && tags.length > 0) {
      removeTag(tags[tags.length - 1]);
    }
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Etiquetas
      </label>
      <div className="relative">
        <div className="flex flex-wrap gap-2 p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 min-h-[42px]">
          {tags.map((tag) => (
            <motion.span
              key={tag}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="inline-flex items-center gap-1 px-2 py-1 bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 rounded-full text-sm"
            >
              <FiTag size={12} />
              {tag}
              <button
                onClick={() => removeTag(tag)}
                className="hover:text-primary-900 dark:hover:text-primary-100"
              >
                <FiX size={14} />
              </button>
            </motion.span>
          ))}
          <input
            type="text"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              setShowSuggestions(true);
            }}
            onKeyDown={handleKeyDown}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
            placeholder="Agregar etiqueta..."
            className="flex-1 min-w-[120px] border-none outline-none bg-transparent text-gray-900 dark:text-white placeholder-gray-400"
          />
        </div>

        <AnimatePresence>
          {showSuggestions && availableSuggestions.length > 0 && inputValue && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 max-h-48 overflow-y-auto"
            >
              {availableSuggestions.slice(0, 5).map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => addTag(suggestion)}
                  className="w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 text-sm text-gray-700 dark:text-gray-300"
                >
                  <FiPlus size={14} className="inline mr-2" />
                  {suggestion}
                </button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}


