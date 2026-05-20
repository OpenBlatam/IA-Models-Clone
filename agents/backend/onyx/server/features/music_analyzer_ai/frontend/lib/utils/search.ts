/**
 * Search utility functions.
 * Provides helper functions for search and filtering operations.
 */

/**
 * Searches text in a string (case-insensitive).
 * @param text - Text to search in
 * @param query - Search query
 * @returns True if text contains query
 */
export function searchText(text: string, query: string): boolean {
  if (!query.trim()) {
    return true;
  }

  return text.toLowerCase().includes(query.toLowerCase());
}

/**
 * Searches text with fuzzy matching.
 * @param text - Text to search in
 * @param query - Search query
 * @returns True if text matches query (fuzzy)
 */
export function fuzzySearch(text: string, query: string): boolean {
  if (!query.trim()) {
    return true;
  }

  const textLower = text.toLowerCase();
  const queryLower = query.toLowerCase();
  let queryIndex = 0;

  for (let i = 0; i < textLower.length && queryIndex < queryLower.length; i++) {
    if (textLower[i] === queryLower[queryIndex]) {
      queryIndex++;
    }
  }

  return queryIndex === queryLower.length;
}

/**
 * Highlights search terms in text.
 * @param text - Text to highlight
 * @param query - Search query
 * @param highlightClass - CSS class for highlight (default: 'highlight')
 * @returns HTML string with highlights
 */
export function highlightSearch(
  text: string,
  query: string,
  highlightClass: string = 'highlight'
): string {
  if (!query.trim()) {
    return text;
  }

  const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
  return text.replace(regex, `<mark class="${highlightClass}">$1</mark>`);
}

/**
 * Escapes special regex characters.
 * @param string - String to escape
 * @returns Escaped string
 */
function escapeRegex(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Filters array by search query.
 * @param items - Array of items
 * @param query - Search query
 * @param getSearchText - Function to get searchable text from item
 * @returns Filtered array
 */
export function filterBySearch<T>(
  items: T[],
  query: string,
  getSearchText: (item: T) => string
): T[] {
  if (!query.trim()) {
    return items;
  }

  return items.filter((item) => {
    const searchText = getSearchText(item);
    return searchText(searchText, query);
  });
}

/**
 * Sorts array by relevance to search query.
 * @param items - Array of items
 * @param query - Search query
 * @param getSearchText - Function to get searchable text from item
 * @returns Sorted array
 */
export function sortByRelevance<T>(
  items: T[],
  query: string,
  getSearchText: (item: T) => string
): T[] {
  if (!query.trim()) {
    return items;
  }

  const queryLower = query.toLowerCase();

  return [...items].sort((a, b) => {
    const textA = getSearchText(a).toLowerCase();
    const textB = getSearchText(b).toLowerCase();

    // Exact match first
    if (textA === queryLower) return -1;
    if (textB === queryLower) return 1;

    // Starts with query
    const startsWithA = textA.startsWith(queryLower);
    const startsWithB = textB.startsWith(queryLower);
    if (startsWithA && !startsWithB) return -1;
    if (startsWithB && !startsWithA) return 1;

    // Contains query
    const containsA = textA.includes(queryLower);
    const containsB = textB.includes(queryLower);
    if (containsA && !containsB) return -1;
    if (containsB && !containsA) return 1;

    // Position of match
    const indexA = textA.indexOf(queryLower);
    const indexB = textB.indexOf(queryLower);
    if (indexA !== -1 && indexB !== -1) {
      return indexA - indexB;
    }

    return 0;
  });
}

/**
 * Gets search suggestions from items.
 * @param items - Array of items
 * @param query - Search query
 * @param getSearchText - Function to get searchable text from item
 * @param maxSuggestions - Maximum number of suggestions (default: 5)
 * @returns Array of suggestions
 */
export function getSearchSuggestions<T>(
  items: T[],
  query: string,
  getSearchText: (item: T) => string,
  maxSuggestions: number = 5
): T[] {
  if (!query.trim()) {
    return [];
  }

  const filtered = filterBySearch(items, query, getSearchText);
  const sorted = sortByRelevance(filtered, query, getSearchText);

  return sorted.slice(0, maxSuggestions);
}

