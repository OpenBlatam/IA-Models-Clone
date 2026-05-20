/**
 * Search utilities
 */

// Simple search in text
export function searchInText(text: string, query: string): boolean {
  return text.toLowerCase().includes(query.toLowerCase());
}

// Search in array of strings
export function searchInArray(array: string[], query: string): string[] {
  const lowerQuery = query.toLowerCase();
  return array.filter((item) => item.toLowerCase().includes(lowerQuery));
}

// Search in object
export function searchInObject<T extends Record<string, any>>(
  obj: T,
  query: string,
  keys?: (keyof T)[]
): boolean {
  const lowerQuery = query.toLowerCase();

  if (keys) {
    return keys.some((key) => {
      const value = String(obj[key]).toLowerCase();
      return value.includes(lowerQuery);
    });
  }

  return Object.values(obj).some((value) => {
    return String(value).toLowerCase().includes(lowerQuery);
  });
}

// Fuzzy search (simple implementation)
export function fuzzySearch(text: string, query: string): boolean {
  const textLower = text.toLowerCase();
  const queryLower = query.toLowerCase();

  let textIndex = 0;
  let queryIndex = 0;

  while (textIndex < textLower.length && queryIndex < queryLower.length) {
    if (textLower[textIndex] === queryLower[queryIndex]) {
      queryIndex++;
    }
    textIndex++;
  }

  return queryIndex === queryLower.length;
}

// Highlight search results
export function highlightSearch(text: string, query: string): string {
  if (!query) return text;

  const regex = new RegExp(`(${query})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
}

// Get search score (simple relevance scoring)
export function getSearchScore(text: string, query: string): number {
  const textLower = text.toLowerCase();
  const queryLower = query.toLowerCase();

  if (textLower === queryLower) return 100;
  if (textLower.startsWith(queryLower)) return 80;
  if (textLower.includes(queryLower)) return 60;
  if (fuzzySearch(text, query)) return 40;

  return 0;
}



