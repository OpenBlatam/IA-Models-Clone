export const search = <T,>(
  array: T[],
  query: string,
  searchFn: (item: T, query: string) => boolean
): T[] => {
  const lowerQuery = query.toLowerCase().trim();
  if (!lowerQuery) return array;
  return array.filter((item) => searchFn(item, lowerQuery));
};

export const searchByFields = <T extends Record<string, unknown>>(
  array: T[],
  query: string,
  fields: (keyof T)[]
): T[] => {
  const lowerQuery = query.toLowerCase().trim();
  if (!lowerQuery) return array;

  return array.filter((item) => {
    return fields.some((field) => {
      const value = item[field];
      if (value === null || value === undefined) return false;
      return String(value).toLowerCase().includes(lowerQuery);
    });
  });
};

export const fuzzySearch = <T,>(
  array: T[],
  query: string,
  getString: (item: T) => string
): T[] => {
  const lowerQuery = query.toLowerCase().trim();
  if (!lowerQuery) return array;

  const queryChars = lowerQuery.split('');

  return array.filter((item) => {
    const str = getString(item).toLowerCase();
    let searchIndex = 0;

    for (let i = 0; i < str.length && searchIndex < queryChars.length; i++) {
      if (str[i] === queryChars[searchIndex]) {
        searchIndex++;
      }
    }

    return searchIndex === queryChars.length;
  });
};

export const searchStartsWith = <T,>(
  array: T[],
  query: string,
  getString: (item: T) => string
): T[] => {
  const lowerQuery = query.toLowerCase().trim();
  if (!lowerQuery) return array;

  return array.filter((item) => {
    return getString(item).toLowerCase().startsWith(lowerQuery);
  });
};

export const searchEndsWith = <T,>(
  array: T[],
  query: string,
  getString: (item: T) => string
): T[] => {
  const lowerQuery = query.toLowerCase().trim();
  if (!lowerQuery) return array;

  return array.filter((item) => {
    return getString(item).toLowerCase().endsWith(lowerQuery);
  });
};

export const searchExact = <T,>(
  array: T[],
  query: string,
  getString: (item: T) => string,
  caseSensitive = false
): T[] => {
  const normalizedQuery = caseSensitive ? query.trim() : query.toLowerCase().trim();
  if (!normalizedQuery) return array;

  return array.filter((item) => {
    const str = getString(item);
    const normalizedStr = caseSensitive ? str.trim() : str.toLowerCase().trim();
    return normalizedStr === normalizedQuery;
  });
};

export const highlightMatches = (
  text: string,
  query: string,
  className = 'bg-yellow-200'
): string => {
  if (!query.trim()) return text;

  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  return text.replace(regex, `<mark class="${className}">$1</mark>`);
};

