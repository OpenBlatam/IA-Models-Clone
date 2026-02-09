'use client';

import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Search } from 'lucide-react';
import type { SearchInputProps } from '@/lib/types/components';

export const SearchInput = ({
  value,
  onChange,
  onSearch,
  placeholder = 'Buscar...',
  disabled = false,
  ariaLabel = 'Buscar',
}: SearchInputProps): JSX.Element => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter') {
      onSearch();
    }
  };

  return (
    <div className="flex gap-4">
      <Input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-1"
        aria-label={ariaLabel}
        disabled={disabled}
      />
      <Button onClick={onSearch} disabled={disabled || !value.trim()} aria-label="Buscar">
        <Search className="h-4 w-4 mr-2" />
        Buscar
      </Button>
    </div>
  );
};

