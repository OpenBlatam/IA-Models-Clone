/**
 * Search and filter utilities
 * @module robot-3d-view/utils/search
 */

/**
 * Search result
 */
export interface SearchResult<T> {
  item: T;
  score: number;
  matches: string[];
}

/**
 * Search options
 */
export interface SearchOptions {
  caseSensitive?: boolean;
  fuzzy?: boolean;
  threshold?: number;
}

/**
 * Searches through items with fuzzy matching
 * 
 * @param query - Search query
 * @param items - Items to search
 * @param getSearchableText - Function to get searchable text from item
 * @param options - Search options
 * @returns Search results sorted by score
 */
export function search<T>(
  query: string,
  items: T[],
  getSearchableText: (item: T) => string,
  options: SearchOptions = {}
): SearchResult<T>[] {
  const {
    caseSensitive = false,
    fuzzy = true,
    threshold = 0.3,
  } = options;

  if (!query.trim()) {
    return items.map((item) => ({
      item,
      score: 1,
      matches: [],
    }));
  }

  const normalizedQuery = caseSensitive ? query : query.toLowerCase();
  const results: SearchResult<T>[] = [];

  for (const item of items) {
    const text = getSearchableText(item);
    const normalizedText = caseSensitive ? text : text.toLowerCase();

    let score = 0;
    const matches: string[] = [];

    if (fuzzy) {
      // Fuzzy matching
      score = fuzzyMatch(normalizedQuery, normalizedText);
    } else {
      // Exact matching
      if (normalizedText.includes(normalizedQuery)) {
        score = 1;
        matches.push(normalizedQuery);
      }
    }

    if (score >= threshold) {
      results.push({ item, score, matches });
    }
  }

  // Sort by score (descending)
  results.sort((a, b) => b.score - a.score);

  return results;
}

/**
 * Fuzzy match algorithm
 * 
 * @param query - Query string
 * @param text - Text to search in
 * @returns Match score (0-1)
 */
function fuzzyMatch(query: string, text: string): number {
  if (query.length === 0) return 1;
  if (text.length === 0) return 0;

  // Exact match
  if (text.includes(query)) {
    return 1;
  }

  // Character matching
  let queryIndex = 0;
  let textIndex = 0;
  let matches = 0;

  while (queryIndex < query.length && textIndex < text.length) {
    if (query[queryIndex] === text[textIndex]) {
      matches++;
      queryIndex++;
    }
    textIndex++;
  }

  if (queryIndex === query.length) {
    // All characters matched
    return matches / query.length;
  }

  return 0;
}

/**
 * Filters items based on criteria
 * 
 * @param items - Items to filter
 * @param predicate - Filter predicate
 * @returns Filtered items
 */
export function filter<T>(
  items: T[],
  predicate: (item: T) => boolean
): T[] {
  return items.filter(predicate);
}

/**
 * Groups items by a key
 * 
 * @param items - Items to group
 * @param getKey - Function to get grouping key
 * @returns Grouped items
 */
export function groupBy<T, K extends string | number>(
  items: T[],
  getKey: (item: T) => K
): Map<K, T[]> {
  const groups = new Map<K, T[]>();

  for (const item of items) {
    const key = getKey(item);
    const group = groups.get(key) || [];
    group.push(item);
    groups.set(key, group);
  }

  return groups;
}

/**
 * Sorts items by a key
 * 
 * @param items - Items to sort
 * @param getKey - Function to get sort key
 * @param direction - Sort direction
 * @returns Sorted items
 */
export function sortBy<T, K>(
  items: T[],
  getKey: (item: T) => K,
  direction: 'asc' | 'desc' = 'asc'
): T[] {
  const sorted = [...items].sort((a, b) => {
    const keyA = getKey(a);
    const keyB = getKey(b);

    if (keyA < keyB) return direction === 'asc' ? -1 : 1;
    if (keyA > keyB) return direction === 'asc' ? 1 : -1;
    return 0;
  });

  return sorted;
}



