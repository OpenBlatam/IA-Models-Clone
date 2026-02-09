/**
 * Select component (enhanced version)
 */

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { ChevronDown, Check } from 'lucide-react';
import { Button } from './Button';

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
  icon?: React.ReactNode;
}

export interface SelectProps {
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  label?: string;
  error?: string;
  id?: string;
}

export const Select: React.FC<SelectProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Seleccionar...',
  disabled = false,
  className,
  label,
  error,
  id,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const selectRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [isOpen]);

  const selectedOption = options.find((opt) => opt.value === value);

  const handleSelect = (optionValue: string) => {
    if (onChange && !options.find((opt) => opt.value === optionValue)?.disabled) {
      onChange(optionValue);
      setIsOpen(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (disabled) {
      return;
    }

    switch (event.key) {
      case 'Enter':
      case ' ':
        event.preventDefault();
        setIsOpen(!isOpen);
        break;
      case 'Escape':
        setIsOpen(false);
        break;
      case 'ArrowDown':
        event.preventDefault();
        if (!isOpen) {
          setIsOpen(true);
        } else {
          const currentIndex = options.findIndex((opt) => opt.value === value);
          const nextIndex = (currentIndex + 1) % options.length;
          if (!options[nextIndex].disabled) {
            onChange?.(options[nextIndex].value);
          }
        }
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (isOpen) {
          const currentIndex = options.findIndex((opt) => opt.value === value);
          const prevIndex = (currentIndex - 1 + options.length) % options.length;
          if (!options[prevIndex].disabled) {
            onChange?.(options[prevIndex].value);
          }
        }
        break;
    }
  };

  const selectId = id || `select-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <label htmlFor={selectId} className="block text-sm font-medium mb-2">
          {label}
        </label>
      )}
      <div ref={selectRef} className="relative">
        <Button
          ref={buttonRef}
          id={selectId}
          variant="outline"
          onClick={() => !disabled && setIsOpen(!isOpen)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          className={cn(
            'w-full justify-between',
            error && 'border-destructive',
            !selectedOption && 'text-muted-foreground'
          )}
          aria-expanded={isOpen}
          aria-haspopup="listbox"
          aria-label={label || 'Select'}
          tabIndex={disabled ? -1 : 0}
        >
          <span className="flex items-center gap-2">
            {selectedOption?.icon && (
              <span aria-hidden="true">{selectedOption.icon}</span>
            )}
            {selectedOption ? selectedOption.label : placeholder}
          </span>
          <ChevronDown
            className={cn(
              'h-4 w-4 transition-transform',
              isOpen && 'transform rotate-180'
            )}
            aria-hidden="true"
          />
        </Button>

        {isOpen && (
          <div
            className="absolute z-50 w-full mt-1 bg-popover border rounded-md shadow-lg max-h-60 overflow-auto"
            role="listbox"
            aria-label={label || 'Options'}
          >
            {options.map((option) => {
              const isSelected = value === option.value;
              const isDisabled = option.disabled;

              return (
                <div
                  key={option.value}
                  onClick={() => handleSelect(option.value)}
                  onKeyDown={(e) => {
                    if (!isDisabled && (e.key === 'Enter' || e.key === ' ')) {
                      e.preventDefault();
                      handleSelect(option.value);
                    }
                  }}
                  className={cn(
                    'flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors',
                    isSelected && 'bg-accent',
                    !isSelected && !isDisabled && 'hover:bg-accent',
                    isDisabled && 'opacity-50 cursor-not-allowed'
                  )}
                  role="option"
                  aria-selected={isSelected}
                  aria-disabled={isDisabled}
                  tabIndex={isDisabled ? -1 : 0}
                >
                  {option.icon && (
                    <span aria-hidden="true">{option.icon}</span>
                  )}
                  <span className="flex-1">{option.label}</span>
                  {isSelected && (
                    <Check className="h-4 w-4 text-primary" aria-hidden="true" />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
      {error && (
        <p className="mt-1 text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};
