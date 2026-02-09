import React, { memo, useCallback, useState, useEffect } from 'react';
import { useDebounce } from '@/hooks';
import { cn } from '@/lib/utils';
import { DEBOUNCE_DELAYS } from '@/lib/constants';

interface SearchInputProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  className?: string;
  debounceMs?: number;
}

const SearchInput = memo(({
  onSearch,
  placeholder = 'Search...',
  className = '',
  debounceMs = DEBOUNCE_DELAYS.SEARCH,
}: SearchInputProps): JSX.Element => {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, debounceMs);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>): void => {
    setQuery(e.target.value);
  }, []);

  const handleClear = useCallback((): void => {
    setQuery('');
    onSearch('');
  }, [onSearch]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Escape') {
      handleClear();
    }
  }, [handleClear]);

  React.useEffect(() => {
    onSearch(debouncedQuery);
  }, [debouncedQuery, onSearch]);

  return (
    <div className={cn('relative', className)}>
      <input
        type="text"
        value={query}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={cn(
          'w-full px-4 py-2 pr-10 border border-gray-300 rounded-lg',
          'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent'
        )}
        aria-label="Search input"
      />
      {query && (
        <button
          type="button"
          onClick={handleClear}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600"
          aria-label="Clear search"
          tabIndex={0}
        >
          ×
        </button>
      )}
    </div>
  );
});

SearchInput.displayName = 'SearchInput';

export default SearchInput;
