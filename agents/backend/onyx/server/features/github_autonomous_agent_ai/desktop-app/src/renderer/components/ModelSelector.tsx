import React, { useState, useRef, useEffect } from 'react';
import { cn } from '../utils/cn';
import { AVAILABLE_MODELS, getModelConfig, type AIModel } from '../lib/ai-providers';

interface ModelSelectorProps {
  selectedModel: AIModel;
  onModelChange: (model: AIModel) => void;
  className?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedModelConfig = getModelConfig(selectedModel);

  const modelsByProvider = {
    deepseek: AVAILABLE_MODELS.filter((m) => m.provider === 'deepseek'),
    openrouter: AVAILABLE_MODELS.filter((m) => m.provider === 'openrouter'),
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className={cn('relative', className)} ref={dropdownRef}>
      <label className="block text-sm font-medium text-black mb-2">
        Modelo de IA
      </label>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'w-full px-4 py-3 bg-white border border-gray-300 rounded-lg text-black',
          'focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent text-sm',
          'flex items-center justify-between',
          'hover:border-gray-400 transition-colors'
        )}
      >
        <div className="flex items-center gap-2">
          <span className="font-medium">
            {selectedModelConfig?.name || selectedModel}
          </span>
          <span className="text-xs text-gray-500">
            ({selectedModelConfig?.provider})
          </span>
        </div>
        <svg
          className={cn(
            'w-4 h-4 transition-transform',
            isOpen && 'rotate-180'
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full mt-2 bg-white border border-gray-300 rounded-lg shadow-lg max-h-64 overflow-y-auto">
          <div className="p-2">
            {/* DeepSeek Models */}
            <div className="mb-2">
              <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase">
                DeepSeek
              </div>
              {modelsByProvider.deepseek.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => {
                    onModelChange(model.id);
                    setIsOpen(false);
                  }}
                  className={cn(
                    'w-full px-3 py-2 text-left text-sm rounded-md transition-colors',
                    selectedModel === model.id
                      ? 'bg-gray-50 text-black font-medium'
                      : 'text-black hover:bg-gray-100'
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{model.name}</div>
                      {model.description && (
                        <div className="text-xs text-gray-500 mt-0.5">
                          {model.description}
                        </div>
                      )}
                    </div>
                    {selectedModel === model.id && (
                      <svg
                        className="w-4 h-4 text-black"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    )}
                  </div>
                </button>
              ))}
            </div>

            {/* OpenRouter Models */}
            <div>
              <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase">
                OpenRouter
              </div>
              {modelsByProvider.openrouter.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => {
                    onModelChange(model.id);
                    setIsOpen(false);
                  }}
                  className={cn(
                    'w-full px-3 py-2 text-left text-sm rounded-md transition-colors',
                    selectedModel === model.id
                      ? 'bg-gray-50 text-black font-medium'
                      : 'text-black hover:bg-gray-100'
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">{model.name}</div>
                      {model.description && (
                        <div className="text-xs text-gray-500 mt-0.5">
                          {model.description}
                        </div>
                      )}
                    </div>
                    {selectedModel === model.id && (
                      <svg
                        className="w-4 h-4 text-black"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

