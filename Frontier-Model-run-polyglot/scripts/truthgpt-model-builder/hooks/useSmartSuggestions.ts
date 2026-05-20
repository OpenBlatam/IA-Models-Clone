/**
 * useSmartSuggestions - Hook Personalizado para Smart Suggestions
 * ===============================================================
 * 
 * Hook modular para gestionar las sugerencias inteligentes del chat
 */

import { useState, useMemo, useCallback } from 'react';

export interface SmartSuggestion {
  text: string;
  confidence: number;
  category?: string;
}

export interface UseSmartSuggestionsOptions {
  enabled?: boolean;
  mode?: 'auto' | 'manual' | 'off';
  maxSuggestions?: number;
  onSuggestionSelect?: (suggestion: string) => void;
}

export interface UseSmartSuggestionsReturn {
  suggestions: SmartSuggestion[];
  showSuggestions: boolean;
  enabled: boolean;
  mode: 'auto' | 'manual' | 'off';
  setShowSuggestions: (show: boolean) => void;
  setEnabled: (enabled: boolean) => void;
  setMode: (mode: 'auto' | 'manual' | 'off') => void;
  generateSuggestions: (input: string) => SmartSuggestion[];
  selectSuggestion: (suggestion: string) => void;
  clearSuggestions: () => void;
}

// Función helper para generar sugerencias básicas
const generateBasicSuggestions = (input: string, maxSuggestions: number = 5): SmartSuggestion[] => {
  const trimmedInput = input.trim().toLowerCase();
  
  if (!trimmedInput || trimmedInput.length < 2) {
    return [];
  }

  // Sugerencias predefinidas basadas en palabras clave comunes
  const commonSuggestions: Record<string, SmartSuggestion[]> = {
    'model': [
      { text: 'Crear un modelo de clasificación', confidence: 0.9, category: 'model' },
      { text: 'Generar modelo de regresión', confidence: 0.8, category: 'model' },
      { text: 'Modelo de análisis de sentimientos', confidence: 0.85, category: 'model' },
    ],
    'clasificar': [
      { text: 'Clasificador de imágenes', confidence: 0.9, category: 'classification' },
      { text: 'Clasificador de texto', confidence: 0.85, category: 'classification' },
    ],
    'analizar': [
      { text: 'Análisis de sentimientos', confidence: 0.9, category: 'analysis' },
      { text: 'Análisis de datos', confidence: 0.8, category: 'analysis' },
    ],
  };

  // Buscar sugerencias basadas en palabras clave
  for (const [keyword, suggestions] of Object.entries(commonSuggestions)) {
    if (trimmedInput.includes(keyword)) {
      return suggestions.slice(0, maxSuggestions);
    }
  }

  // Sugerencias genéricas si no hay coincidencias
  return [
    { text: 'Crear un modelo de IA', confidence: 0.7, category: 'generic' },
    { text: 'Generar modelo TruthGPT', confidence: 0.7, category: 'generic' },
  ].slice(0, maxSuggestions);
};

export const useSmartSuggestions = (
  options: UseSmartSuggestionsOptions = {}
): UseSmartSuggestionsReturn => {
  const {
    enabled: initialEnabled = true,
    mode: initialMode = 'auto',
    maxSuggestions = 5,
    onSuggestionSelect,
  } = options;

  const [suggestions, setSuggestions] = useState<SmartSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [enabled, setEnabled] = useState(initialEnabled);
  const [mode, setMode] = useState<'auto' | 'manual' | 'off'>(initialMode);

  const generateSuggestions = useCallback(
    (input: string): SmartSuggestion[] => {
      if (!enabled || mode === 'off' || !input.trim()) {
        return [];
      }

      const newSuggestions = generateBasicSuggestions(input, maxSuggestions);
      setSuggestions(newSuggestions);
      
      // Auto-show si está en modo auto y hay sugerencias
      if (mode === 'auto' && newSuggestions.length > 0) {
        setShowSuggestions(true);
      }

      return newSuggestions;
    },
    [enabled, mode, maxSuggestions]
  );

  const selectSuggestion = useCallback(
    (suggestion: string) => {
      if (onSuggestionSelect) {
        onSuggestionSelect(suggestion);
      }
      setShowSuggestions(false);
      setSuggestions([]);
    },
    [onSuggestionSelect]
  );

  const clearSuggestions = useCallback(() => {
    setSuggestions([]);
    setShowSuggestions(false);
  }, []);

  return {
    suggestions,
    showSuggestions,
    enabled,
    mode,
    setShowSuggestions,
    setEnabled,
    setMode,
    generateSuggestions,
    selectSuggestion,
    clearSuggestions,
  };
};


