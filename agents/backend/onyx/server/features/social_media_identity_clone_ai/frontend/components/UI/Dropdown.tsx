import { useState, useRef, useCallback } from 'react';
import { useClickOutside } from '@/hooks';
import { cn } from '@/lib/utils';

interface DropdownOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface DropdownProps {
  options: DropdownOption[];
  value?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

const Dropdown = ({
  options,
  value,
  onChange,
  placeholder = 'Select an option',
  className = '',
  disabled = false,
}: DropdownProps): JSX.Element => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useClickOutside(dropdownRef, () => setIsOpen(false));

  const handleToggle = useCallback((): void => {
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  }, [disabled, isOpen]);

  const handleSelect = useCallback(
    (optionValue: string): void => {
      if (!disabled) {
        onChange(optionValue);
        setIsOpen(false);
      }
    },
    [disabled, onChange]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>): void => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleToggle();
      } else if (e.key === 'Escape') {
        setIsOpen(false);
      }
    },
    [handleToggle]
  );

  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div ref={dropdownRef} className={cn('relative', className)}>
      <button
        type="button"
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={cn(
          'w-full px-4 py-2 text-left border border-gray-300 rounded-lg',
          'focus:outline-none focus:ring-2 focus:ring-primary-500',
          disabled && 'opacity-50 cursor-not-allowed',
          'flex items-center justify-between'
        )}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label={selectedOption?.label || placeholder}
      >
        <span className={cn(!selectedOption && 'text-gray-500')}>
          {selectedOption?.label || placeholder}
        </span>
        <span className="text-gray-400" aria-hidden="true">
          {isOpen ? '▲' : '▼'}
        </span>
      </button>
      {isOpen && (
        <div
          className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-auto"
          role="listbox"
        >
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleSelect(option.value)}
              disabled={option.disabled}
              className={cn(
                'w-full px-4 py-2 text-left hover:bg-gray-100 transition-colors',
                value === option.value && 'bg-primary-50 text-primary-700',
                option.disabled && 'opacity-50 cursor-not-allowed'
              )}
              role="option"
              aria-selected={value === option.value}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default Dropdown;



