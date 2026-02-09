/**
 * Search utilities
 */

/**
 * Simple search - case insensitive
 */
export const simpleSearch = (text: string, searchTerm: string): boolean => {
  return text.toLowerCase().includes(searchTerm.toLowerCase());
};

/**
 * Fuzzy search
 */
export const fuzzySearch = (text: string, searchTerm: string): boolean => {
  const textLower = text.toLowerCase();
  const searchLower = searchTerm.toLowerCase();
  let searchIndex = 0;

  for (let i = 0; i < textLower.length; i++) {
    if (textLower[i] === searchLower[searchIndex]) {
      searchIndex++;
      if (searchIndex === searchLower.length) {
        return true;
      }
    }
  }

  return false;
};

/**
 * Highlight search term in text
 */
export const highlightSearchTerm = (
  text: string,
  searchTerm: string
): Array<{ text: string; highlight: boolean }> => {
  if (!searchTerm.trim()) {
    return [{ text, highlight: false }];
  }

  const parts: Array<{ text: string; highlight: boolean }> = [];
  const lowerText = text.toLowerCase();
  const lowerSearchTerm = searchTerm.toLowerCase();
  let lastIndex = 0;
  let index = lowerText.indexOf(lowerSearchTerm, lastIndex);

  while (index !== -1) {
    if (index > lastIndex) {
      parts.push({
        text: text.substring(lastIndex, index),
        highlight: false,
      });
    }

    parts.push({
      text: text.substring(index, index + searchTerm.length),
      highlight: true,
    });

    lastIndex = index + searchTerm.length;
    index = lowerText.indexOf(lowerSearchTerm, lastIndex);
  }

  if (lastIndex < text.length) {
    parts.push({
      text: text.substring(lastIndex),
      highlight: false,
    });
  }

  return parts;
};

/**
 * Search with scoring
 */
export const searchWithScore = (
  text: string,
  searchTerm: string
): { match: boolean; score: number } => {
  if (!searchTerm.trim()) {
    return { match: true, score: 0 };
  }

  const textLower = text.toLowerCase();
  const searchLower = searchTerm.toLowerCase();

  // Exact match
  if (textLower === searchLower) {
    return { match: true, score: 100 };
  }

  // Starts with
  if (textLower.startsWith(searchLower)) {
    return { match: true, score: 80 };
  }

  // Contains
  if (textLower.includes(searchLower)) {
    return { match: true, score: 60 };
  }

  // Fuzzy match
  if (fuzzySearch(text, searchTerm)) {
    return { match: true, score: 40 };
  }

  return { match: false, score: 0 };
};

/**
 * Search multiple fields
 */
export const searchMultipleFields = <T,>(
  item: T,
  searchTerm: string,
  fields: (keyof T)[]
): boolean => {
  return fields.some((field) => {
    const value = item[field];
    return simpleSearch(String(value || ''), searchTerm);
  });
};

