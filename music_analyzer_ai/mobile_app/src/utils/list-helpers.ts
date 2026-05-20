/**
 * Utility functions for working with lists
 */

/**
 * Chunks an array into smaller arrays of specified size
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

/**
 * Groups array items by a key function
 */
export function groupBy<T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> {
  return array.reduce((groups, item) => {
    const key = keyFn(item);
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(item);
    return groups;
  }, {} as Record<K, T[]>);
}

/**
 * Creates a stable sort function
 */
export function createStableSort<T>(
  compareFn: (a: T, b: T) => number
): (a: T, b: T) => number {
  return (a: T, b: T) => {
    const result = compareFn(a, b);
    if (result !== 0) return result;
    // Stable sort: maintain original order for equal items
    return 0;
  };
}

/**
 * Creates a search function for text matching
 */
export function createTextSearch<T>(
  getText: (item: T) => string,
  caseSensitive = false
): (item: T, query: string) => boolean {
  return (item: T, query: string) => {
    const text = getText(item);
    const searchText = caseSensitive ? text : text.toLowerCase();
    const searchQuery = caseSensitive ? query : query.toLowerCase();
    return searchText.includes(searchQuery);
  };
}

/**
 * Creates a multi-field search function
 */
export function createMultiFieldSearch<T>(
  fields: Array<(item: T) => string>,
  caseSensitive = false
): (item: T, query: string) => boolean {
  return (item: T, query: string) => {
    const searchQuery = caseSensitive ? query : query.toLowerCase();
    return fields.some((field) => {
      const text = field(item);
      const searchText = caseSensitive ? text : text.toLowerCase();
      return searchText.includes(searchQuery);
    });
  };
}

/**
 * Calculates pagination info
 */
export function getPaginationInfo(
  totalItems: number,
  pageSize: number,
  currentPage: number
) {
  const totalPages = Math.ceil(totalItems / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, totalItems);
  const hasNext = currentPage < totalPages;
  const hasPrevious = currentPage > 1;

  return {
    totalPages,
    startIndex,
    endIndex,
    hasNext,
    hasPrevious,
    currentPage,
    pageSize,
    totalItems,
  };
}

/**
 * Creates a paginated data getter
 */
export function paginate<T>(items: T[], page: number, pageSize: number): T[] {
  const startIndex = (page - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  return items.slice(startIndex, endIndex);
}

