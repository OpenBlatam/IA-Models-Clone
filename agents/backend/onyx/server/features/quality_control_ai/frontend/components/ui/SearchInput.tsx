'use client';

import { memo, useState, useCallback, type ChangeEvent } from 'react';
import { Search, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Input } from './Input';

interface SearchInputProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  className?: string;
  debounceMs?: number;
  onClear?: () => void;
  autoFocus?: boolean;
}

const SearchInput = memo(
  ({
    value: controlledValue,
    onChange,
    placeholder = 'Search...',
    className,
    debounceMs = 300,
    onClear,
    autoFocus = false,
  }: SearchInputProps): JSX.Element => {
    const [internalValue, setInternalValue] = useState('');
    const isControlled = controlledValue !== undefined;
    const value = isControlled ? controlledValue : internalValue;

    const handleChange = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        if (!isControlled) {
          setInternalValue(newValue);
        }
        onChange?.(newValue);
      },
      [isControlled, onChange]
    );

    const handleClear = useCallback(() => {
      if (!isControlled) {
        setInternalValue('');
      }
      onChange?.('');
      onClear?.();
    }, [isControlled, onChange, onClear]);

    return (
      <div className={cn('relative', className)}>
        <Search
          className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400"
          aria-hidden="true"
        />
        <Input
          type="text"
          value={value}
          onChange={handleChange}
          placeholder={placeholder}
          className="pl-10 pr-10"
          autoFocus={autoFocus}
          aria-label="Search input"
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Clear search"
          >
            <X className="w-4 h-4" aria-hidden="true" />
          </button>
        )}
      </div>
    );
  }
);

SearchInput.displayName = 'SearchInput';

export default SearchInput;

