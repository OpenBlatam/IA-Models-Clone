'use client';

import { cn } from '@/lib/utils';

interface Option {
  value: string | number;
  label: string;
}

interface MultiSelectButtonsProps {
  options: Option[];
  selected: (string | number)[];
  onChange: (value: string | number) => void;
  label?: string;
  error?: string;
  required?: boolean;
}

const MultiSelectButtons = ({ options, selected, onChange, label, error, required }: MultiSelectButtonsProps) => {
  const handleToggle = (value: string | number) => {
    onChange(value);
  };

  const handleKeyDown = (e: React.KeyboardEvent, value: string | number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleToggle(value);
    }
  };

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const isSelected = selected.includes(option.value);
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => handleToggle(option.value)}
              onKeyDown={(e) => handleKeyDown(e, option.value)}
              className={cn(
                'px-4 py-2 rounded-lg border transition-colors',
                isSelected
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
              )}
              tabIndex={0}
              aria-label={`Seleccionar ${option.label}`}
              aria-pressed={isSelected}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      {error && (
        <p className="mt-1 text-sm text-red-600" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};

export { MultiSelectButtons };

