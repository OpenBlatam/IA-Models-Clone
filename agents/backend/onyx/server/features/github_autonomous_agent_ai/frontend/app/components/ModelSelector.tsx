'use client';

import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { AVAILABLE_MODELS, AIModel, ModelConfig, getModelConfig } from '../lib/ai-providers';
import { cn } from '../utils/cn';

interface ModelSelectorProps {
  selectedModel: AIModel;
  onModelChange: (model: AIModel) => void;
  className?: string;
}

export default function ModelSelector({ selectedModel, onModelChange, className }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const selectedModelConfig = useMemo(() => getModelConfig(selectedModel), [selectedModel]);

  const modelsByProvider = useMemo(() => ({
    deepseek: AVAILABLE_MODELS.filter(m => m.provider === 'deepseek'),
    openrouter: AVAILABLE_MODELS.filter(m => m.provider === 'openrouter'),
  }), []);

  const handleToggle = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  const handleModelSelect = useCallback((model: AIModel) => {
    onModelChange(model);
    setIsOpen(false);
    buttonRef.current?.focus();
  }, [onModelChange]);

  // Close dropdown on outside click and escape key
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        isOpen &&
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        !buttonRef.current?.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        buttonRef.current?.focus();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  return (
    <div className={cn("relative", className)}>
      <label htmlFor="model-selector-button" className="block text-black text-sm mb-2 font-medium">
        Modelo de IA
      </label>
      <button
        ref={buttonRef}
        id="model-selector-button"
        type="button"
        onClick={handleToggle}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-controls="model-selector-listbox"
        aria-label={`Modelo seleccionado: ${selectedModelConfig?.name || selectedModel}`}
        className={cn(
          "w-full px-4 py-3 bg-white border rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent text-sm",
          "flex items-center justify-between",
          "hover:border-gray-400 transition-colors"
        )}
      >
        <div className="flex items-center gap-2">
          <span className="font-medium">{selectedModelConfig?.name || selectedModel}</span>
          <span className="text-xs text-gray-500">({selectedModelConfig?.provider})</span>
        </div>
        <svg
          className={cn("w-4 h-4 transition-transform", isOpen && "rotate-180")}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
            aria-hidden="true"
          />
          <div 
            ref={dropdownRef}
            id="model-selector-listbox"
            role="listbox"
            aria-label="Lista de modelos disponibles"
            className="absolute z-20 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-96 overflow-y-auto"
          >
            {/* DeepSeek Models */}
            {modelsByProvider.deepseek.length > 0 && (
              <div className="p-2" role="group" aria-label="Modelos DeepSeek">
                <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase">
                  DeepSeek
                </div>
                {modelsByProvider.deepseek.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    role="option"
                    aria-selected={selectedModel === model.id}
                    onClick={() => handleModelSelect(model.id)}
                    className={cn(
                      "w-full px-3 py-2 text-left text-sm rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2",
                      selectedModel === model.id
                        ? "bg-black text-white"
                        : "hover:bg-gray-100 text-gray-700"
                    )}
                  >
                    <div className="font-medium">{model.name}</div>
                    {model.description && (
                      <div className="text-xs text-gray-500 mt-0.5">{model.description}</div>
                    )}
                  </button>
                ))}
              </div>
            )}

            {/* OpenRouter Models */}
            {modelsByProvider.openrouter.length > 0 && (
              <div 
                className={cn("p-2", modelsByProvider.deepseek.length > 0 && "border-t border-gray-200")}
                role="group"
                aria-label="Modelos OpenRouter"
              >
                <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase">
                  OpenRouter
                </div>
                {modelsByProvider.openrouter.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    role="option"
                    aria-selected={selectedModel === model.id}
                    onClick={() => handleModelSelect(model.id)}
                    className={cn(
                      "w-full px-3 py-2 text-left text-sm rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2",
                      selectedModel === model.id
                        ? "bg-black text-white"
                        : "hover:bg-gray-100 text-gray-700"
                    )}
                  >
                    <div className="font-medium">{model.name}</div>
                    {model.description && (
                      <div className="text-xs text-gray-500 mt-0.5">{model.description}</div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

