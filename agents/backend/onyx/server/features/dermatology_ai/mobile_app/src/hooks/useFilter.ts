import { useState, useMemo } from 'react';

type FilterFunction<T> = (item: T) => boolean;

export const useFilter = <T,>(
  data: T[],
  initialFilter?: FilterFunction<T>
) => {
  const [filterFn, setFilterFn] = useState<FilterFunction<T> | null>(
    initialFilter || null
  );

  const filteredData = useMemo(() => {
    if (!filterFn) return data;
    return data.filter(filterFn);
  }, [data, filterFn]);

  const setFilter = (filter: FilterFunction<T> | null) => {
    setFilterFn(() => filter);
  };

  const clearFilter = () => {
    setFilterFn(null);
  };

  return {
    filteredData,
    filterFn,
    setFilter,
    clearFilter,
    isFiltered: filterFn !== null,
  };
};

