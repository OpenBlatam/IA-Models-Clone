'use client';

import { InputHTMLAttributes, useState } from 'react';
import { FiSearch, FiX } from 'react-icons/fi';
import { Input } from './Input';
import { Button } from './Button';
import { cn } from '@/utils/classNames';

interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
  onSearch?: (value: string) => void;
  onClear?: () => void;
  debounceMs?: number;
  showClearButton?: boolean;
}

export function SearchInput({
  onSearch,
  onClear,
  debounceMs = 300,
  showClearButton = true,
  className,
  ...props
}: SearchInputProps) {
  const [value, setValue] = useState(props.value || '');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setValue(newValue);
    
    if (onSearch) {
      const timeoutId = setTimeout(() => {
        onSearch(newValue);
      }, debounceMs);
      
      return () => clearTimeout(timeoutId);
    }
  };

  const handleClear = () => {
    setValue('');
    onClear?.();
    onSearch?.('');
  };

  return (
    <div className={cn('relative', className)}>
      <Input
        {...props}
        value={value}
        onChange={handleChange}
        leftIcon={<FiSearch size={18} />}
        className="pr-10"
      />
      {showClearButton && value && (
        <button
          onClick={handleClear}
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          aria-label="Limpiar búsqueda"
        >
          <FiX size={18} />
        </button>
      )}
    </div>
  );
}

