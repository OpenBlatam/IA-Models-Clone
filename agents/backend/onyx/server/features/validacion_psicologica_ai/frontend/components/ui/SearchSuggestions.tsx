/**
 * Search suggestions component
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils/cn';
import { Search, X } from 'lucide-react';
import { Button } from './Button';

export interface SearchSuggestion {
  id: string;
  label: string;
  description?: string;
  category?: string;
}

export interface SearchSuggestionsProps {
  suggestions: SearchSuggestion[];
  onSelect: (suggestion: SearchSuggestion) => void;
  onClear?: () => void;
  className?: string;
  maxHeight?: string;
}

export const SearchSuggestions: React.FC<SearchSuggestionsProps> = ({
  suggestions,
  onSelect,
  onClear,
  className,
  maxHeight = 'max-h-64',
}) => {
  if (suggestions.length === 0) {
    return null;
  }

  const handleSelect = (suggestion: SearchSuggestion) => {
    onSelect(suggestion);
  };

  const handleKeyDown = (event: React.KeyboardEvent, suggestion: SearchSuggestion) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleSelect(suggestion);
    }
  };

  const groupedSuggestions = React.useMemo(() => {
    const groups: Record<string, SearchSuggestion[]> = {};
    suggestions.forEach((suggestion) => {
      const category = suggestion.category || 'Otros';
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(suggestion);
    });
    return groups;
  }, [suggestions]);

  return (
    <div
      className={cn(
        'absolute z-50 w-full mt-1 bg-background border rounded-lg shadow-lg overflow-hidden',
        maxHeight,
        'overflow-y-auto',
        className
      )}
      role="listbox"
      aria-label="Sugerencias de búsqueda"
    >
      {onClear && (
        <div className="flex items-center justify-between p-2 border-b">
          <span className="text-xs text-muted-foreground">
            {suggestions.length} sugerencia{suggestions.length !== 1 ? 's' : ''}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClear();
              }
            }}
            aria-label="Limpiar sugerencias"
            tabIndex={0}
          >
            <X className="h-3 w-3" aria-hidden="true" />
          </Button>
        </div>
      )}

      <div className="py-1">
        {Object.entries(groupedSuggestions).map(([category, items]) => (
          <div key={category}>
            {category !== 'Otros' && (
              <div className="px-3 py-1 text-xs font-semibold text-muted-foreground uppercase">
                {category}
              </div>
            )}
            {items.map((suggestion) => (
              <div
                key={suggestion.id}
                className="flex items-start gap-3 px-3 py-2 hover:bg-accent cursor-pointer transition-colors"
                onClick={() => handleSelect(suggestion)}
                onKeyDown={(e) => handleKeyDown(e, suggestion)}
                role="option"
                aria-selected="false"
                tabIndex={0}
              >
                <Search className="h-4 w-4 mt-0.5 text-muted-foreground flex-shrink-0" aria-hidden="true" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{suggestion.label}</div>
                  {suggestion.description && (
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {suggestion.description}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};



