import { useState, useMemo } from 'react';

interface UseSearchOptions<T> {
  data: T[];
  searchFields: (keyof T)[];
  searchTerm?: string;
}

export const useSearch = <T extends Record<string, any>>({
  data,
  searchFields,
  searchTerm = '',
}: UseSearchOptions<T>) => {
  const [search, setSearch] = useState(searchTerm);

  const filteredData = useMemo(() => {
    if (!search.trim()) {
      return data;
    }

    const lowerSearch = search.toLowerCase();

    return data.filter((item) => {
      return searchFields.some((field) => {
        const value = item[field];
        if (value === null || value === undefined) {
          return false;
        }
        return String(value).toLowerCase().includes(lowerSearch);
      });
    });
  }, [data, searchFields, search]);

  return {
    search,
    setSearch,
    filteredData,
  };
};

