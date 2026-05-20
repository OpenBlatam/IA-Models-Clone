/**
 * Filter utilities
 */

// Filter array by key-value
export function filterBy<T>(array: T[], key: keyof T, value: any): T[] {
  return array.filter((item) => item[key] === value);
}

// Filter array by function
export function filterByFunction<T>(array: T[], fn: (item: T) => boolean): T[] {
  return array.filter(fn);
}

// Filter array by multiple conditions
export function filterByMultiple<T>(
  array: T[],
  conditions: Array<{ key: keyof T; value: any }>
): T[] {
  return array.filter((item) => {
    return conditions.every((condition) => item[condition.key] === condition.value);
  });
}

// Filter array by search term
export function filterBySearch<T>(
  array: T[],
  searchTerm: string,
  keys?: (keyof T)[]
): T[] {
  if (!searchTerm) return array;

  const term = searchTerm.toLowerCase();

  return array.filter((item) => {
    if (keys) {
      return keys.some((key) => {
        const value = String(item[key]).toLowerCase();
        return value.includes(term);
      });
    }

    return Object.values(item).some((value) => {
      return String(value).toLowerCase().includes(term);
    });
  });
}

// Filter array by range
export function filterByRange<T>(
  array: T[],
  key: keyof T,
  min: number,
  max: number
): T[] {
  return array.filter((item) => {
    const value = Number(item[key]);
    return value >= min && value <= max;
  });
}

// Filter unique by key
export function filterUniqueBy<T>(array: T[], key: keyof T): T[] {
  const seen = new Set();
  return array.filter((item) => {
    const value = item[key];
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
}



