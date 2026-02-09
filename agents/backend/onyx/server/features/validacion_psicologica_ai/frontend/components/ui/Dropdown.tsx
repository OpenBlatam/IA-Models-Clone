/**
 * Dropdown menu component with accessibility
 */

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils/cn';
import { ChevronDown } from 'lucide-react';

export interface DropdownOption {
  value: string;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export interface DropdownProps {
  options: DropdownOption[];
  value?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  className?: string;
  disabled?: boolean;
}

const Dropdown: React.FC<DropdownProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Seleccionar...',
  label,
  className,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
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

  useEffect(() => {
    if (isOpen && buttonRef.current) {
      buttonRef.current.focus();
    }
  }, [isOpen]);

  const selectedOption = options.find((opt) => opt.value === value);

  const handleToggle = () => {
    if (disabled) {
      return;
    }
    setIsOpen(!isOpen);
  };

  const handleSelect = (optionValue: string) => {
    if (options.find((opt) => opt.value === optionValue)?.disabled) {
      return;
    }
    onChange(optionValue);
    setIsOpen(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) {
      return;
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleToggle();
    } else if (event.key === 'Escape') {
      setIsOpen(false);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      if (!isOpen) {
        setIsOpen(true);
      } else {
        const currentIndex = options.findIndex((opt) => opt.value === value);
        const nextIndex = Math.min(currentIndex + 1, options.length - 1);
        handleSelect(options[nextIndex].value);
      }
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (isOpen) {
        const currentIndex = options.findIndex((opt) => opt.value === value);
        const prevIndex = Math.max(currentIndex - 1, 0);
        handleSelect(options[prevIndex].value);
      }
    }
  };

  return (
    <div ref={dropdownRef} className={cn('relative w-full', className)}>
      {label && (
        <label className="block text-sm font-medium mb-1 text-foreground">{label}</label>
      )}
      <button
        ref={buttonRef}
        type="button"
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={cn(
          'flex items-center justify-between w-full px-3 py-2 text-sm bg-background border border-input rounded-md',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          isOpen && 'ring-2 ring-ring'
        )}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label={label || 'Dropdown menu'}
        tabIndex={disabled ? -1 : 0}
      >
        <span className={cn('truncate', !selectedOption && 'text-muted-foreground')}>
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <ChevronDown
          className={cn('h-4 w-4 text-muted-foreground transition-transform', isOpen && 'rotate-180')}
          aria-hidden="true"
        />
      </button>

      {isOpen && (
        <div
          className="absolute z-50 w-full mt-1 bg-background border border-input rounded-md shadow-lg max-h-60 overflow-auto"
          role="listbox"
        >
          {options.map((option) => {
            const isSelected = value === option.value;
            const isDisabled = option.disabled;

            return (
              <button
                key={option.value}
                type="button"
                onClick={() => handleSelect(option.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleSelect(option.value);
                  }
                }}
                disabled={isDisabled}
                className={cn(
                  'w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors',
                  'focus-visible:outline-none focus-visible:bg-accent focus-visible:text-accent-foreground',
                  isSelected && 'bg-accent text-accent-foreground',
                  isDisabled && 'opacity-50 cursor-not-allowed',
                  !isDisabled && !isSelected && 'hover:bg-accent hover:text-accent-foreground'
                )}
                role="option"
                aria-selected={isSelected}
                tabIndex={isDisabled ? -1 : 0}
              >
                {option.icon && <span aria-hidden="true">{option.icon}</span>}
                <span>{option.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export { Dropdown };




