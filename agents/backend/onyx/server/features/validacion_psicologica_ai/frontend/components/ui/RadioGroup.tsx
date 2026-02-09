/**
 * Radio group component
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils/cn';

export interface RadioOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

export interface RadioGroupProps {
  options: RadioOption[];
  value: string;
  onChange: (value: string) => void;
  name: string;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
}

export const RadioGroup: React.FC<RadioGroupProps> = ({
  options,
  value,
  onChange,
  name,
  className,
  orientation = 'vertical',
}) => {
  const handleChange = (optionValue: string) => {
    if (!options.find((opt) => opt.value === optionValue)?.disabled) {
      onChange(optionValue);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent, optionValue: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleChange(optionValue);
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
      event.preventDefault();
      const currentIndex = options.findIndex((opt) => opt.value === value);
      const nextIndex = (currentIndex + 1) % options.length;
      if (!options[nextIndex].disabled) {
        onChange(options[nextIndex].value);
      }
    } else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
      event.preventDefault();
      const currentIndex = options.findIndex((opt) => opt.value === value);
      const prevIndex = (currentIndex - 1 + options.length) % options.length;
      if (!options[prevIndex].disabled) {
        onChange(options[prevIndex].value);
      }
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label={name}
      className={cn(
        'space-y-2',
        orientation === 'horizontal' && 'flex flex-row gap-4',
        className
      )}
    >
      {options.map((option) => {
        const isSelected = value === option.value;
        const isDisabled = option.disabled;

        return (
          <div
            key={option.value}
            className={cn(
              'flex items-start gap-2 p-3 border rounded-lg cursor-pointer transition-colors',
              isSelected && 'border-primary bg-primary/5',
              !isSelected && 'hover:bg-accent',
              isDisabled && 'opacity-50 cursor-not-allowed'
            )}
            onClick={() => handleChange(option.value)}
            onKeyDown={(e) => handleKeyDown(e, option.value)}
            role="radio"
            aria-checked={isSelected}
            aria-disabled={isDisabled}
            tabIndex={isDisabled ? -1 : 0}
          >
            <div className="flex items-center mt-0.5">
              <div
                className={cn(
                  'h-4 w-4 rounded-full border-2 flex items-center justify-center transition-colors',
                  isSelected
                    ? 'border-primary'
                    : 'border-muted-foreground',
                  isDisabled && 'opacity-50'
                )}
              >
                {isSelected && (
                  <div className="h-2 w-2 rounded-full bg-primary" />
                )}
              </div>
            </div>
            <div className="flex-1">
              <div
                className={cn(
                  'text-sm font-medium',
                  isSelected && 'text-primary',
                  isDisabled && 'opacity-50'
                )}
              >
                {option.label}
              </div>
              {option.description && (
                <div className="text-xs text-muted-foreground mt-1">
                  {option.description}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};



