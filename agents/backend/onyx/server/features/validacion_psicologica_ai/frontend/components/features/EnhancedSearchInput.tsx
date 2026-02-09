/**
 * Enhanced search input with suggestions
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { SearchInput, SearchSuggestions } from '@/components/ui';
import type { SearchSuggestion } from '@/components/ui/SearchSuggestions';
import { useValidations } from '@/hooks/useValidations';
import { useDebounce } from '@/hooks/useDebounce';

export interface EnhancedSearchInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  onSelect?: (suggestion: SearchSuggestion) => void;
  className?: string;
}

export const EnhancedSearchInput: React.FC<EnhancedSearchInputProps> = ({
  value,
  onChange,
  placeholder = 'Buscar...',
  onSelect,
  className,
}) => {
  const { data: validations } = useValidations();
  const [suggestions, setSuggestions] = useState<SearchSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const debouncedValue = useDebounce(value, 300);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!debouncedValue || debouncedValue.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    if (!validations) {
      return;
    }

    const searchLower = debouncedValue.toLowerCase();
    const newSuggestions: SearchSuggestion[] = [];

    validations.forEach((validation) => {
      if (validation.id.toLowerCase().includes(searchLower)) {
        newSuggestions.push({
          id: validation.id,
          label: `Validación ${validation.id.slice(0, 8)}`,
          description: `Estado: ${validation.status}`,
          category: 'Validaciones',
        });
      }

      validation.connected_platforms.forEach((platform) => {
        if (platform.toLowerCase().includes(searchLower)) {
          const existing = newSuggestions.find((s) => s.label === platform);
          if (!existing) {
            newSuggestions.push({
              id: `platform-${platform}`,
              label: platform,
              description: 'Plataforma de redes sociales',
              category: 'Plataformas',
            });
          }
        }
      });
    });

    setSuggestions(newSuggestions.slice(0, 10));
    setShowSuggestions(newSuggestions.length > 0);
  }, [debouncedValue, validations]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSelect = (suggestion: SearchSuggestion) => {
    onChange(suggestion.label);
    setShowSuggestions(false);
    if (onSelect) {
      onSelect(suggestion);
    }
  };

  const handleFocus = () => {
    if (suggestions.length > 0) {
      setShowSuggestions(true);
    }
  };

  return (
    <div ref={containerRef} className={`relative ${className || ''}`}>
      <SearchInput
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        onFocus={handleFocus}
      />
      {showSuggestions && suggestions.length > 0 && (
        <SearchSuggestions
          suggestions={suggestions}
          onSelect={handleSelect}
          onClear={() => {
            setSuggestions([]);
            setShowSuggestions(false);
          }}
        />
      )}
    </div>
  );
};



