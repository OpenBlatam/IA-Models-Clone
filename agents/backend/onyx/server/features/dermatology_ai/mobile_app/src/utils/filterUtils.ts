/**
 * Filter utilities
 */

/**
 * Filter array by search term
 */
export const filterBySearch = <T,>(
  array: T[],
  searchTerm: string,
  keys?: (keyof T)[]
): T[] => {
  if (!searchTerm.trim()) return array;

  const lowerSearchTerm = searchTerm.toLowerCase();

  return array.filter((item) => {
    if (keys) {
      return keys.some((key) => {
        const value = item[key];
        return String(value).toLowerCase().includes(lowerSearchTerm);
      });
    }

    return Object.values(item).some((value) =>
      String(value).toLowerCase().includes(lowerSearchTerm)
    );
  });
};

/**
 * Filter array by multiple conditions
 */
export const filterByMultiple = <T,>(
  array: T[],
  filters: Array<(item: T) => boolean>
): T[] => {
  return array.filter((item) => filters.every((filter) => filter(item)));
};

/**
 * Filter array by date range
 */
export const filterByDateRange = <T,>(
  array: T[],
  dateKey: keyof T,
  startDate: Date,
  endDate: Date
): T[] => {
  return array.filter((item) => {
    const dateValue = item[dateKey];
    if (!dateValue) return false;

    const date = dateValue instanceof Date ? dateValue : new Date(String(dateValue));
    return date >= startDate && date <= endDate;
  });
};

/**
 * Filter array by numeric range
 */
export const filterByNumericRange = <T,>(
  array: T[],
  key: keyof T,
  min: number,
  max: number
): T[] => {
  return array.filter((item) => {
    const value = item[key];
    if (typeof value !== 'number') return false;
    return value >= min && value <= max;
  });
};

/**
 * Filter array by exact match
 */
export const filterByExactMatch = <T,>(
  array: T[],
  key: keyof T,
  value: any
): T[] => {
  return array.filter((item) => item[key] === value);
};

/**
 * Filter array by includes
 */
export const filterByIncludes = <T,>(
  array: T[],
  key: keyof T,
  values: any[]
): T[] => {
  return array.filter((item) => values.includes(item[key]));
};

